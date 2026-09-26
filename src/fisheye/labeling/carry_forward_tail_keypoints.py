"""Carry keypoint edits stranded in older tail-review versions into the newest one.

Before tasks were superseded, a labeler could keep applying keypoint edits to an
older version after a mask Apply had already copied it into a newer version.
Those edits are intact in the older run but absent from the newest one.

For each lineage this tool finds the newest keypoint version, and for each row
the latest applied manual edit anywhere in the lineage (see
``tail_successor_lineage.stranded_keypoint_rows``). Carryable rows are staged
as ordinary checkpoints in the newest version's task, each recording the
checkpoints it replays under ``carried_from``, and applied through the normal
batch keypoint Apply. Older-version tasks are then superseded, and untouched
mask successors of completed mask reviews are retired.

Dry run by default; ``--apply`` writes. Close the affected tasks in the browser
first: opening the carry session closes other sessions on the same task.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import uuid

import numpy as np
import zarr

from fisheye.labeling.assignment_store import (
    LABELER_START_TASK_STATES,
    TASK_SUPERSEDED_STATE,
    LabelingStore,
)
from fisheye.labeling.tail_successor_lineage import (
    KEYPOINT_FAMILY,
    MASK_FAMILY,
    checkpoint_edit_times,
    lineage_tasks,
    read_family_lineage,
    stranded_keypoint_rows,
    task_archive,
    task_run,
)

CARRY_SCHEMA = "palette.labeling.tail_keypoint_carry_forward.v1"


class _RuntimeState:
    """The attribute the server's runtime cache expects."""

    def __init__(self) -> None:
        self.keypoint_sessions: dict[str, object] = {}


def _open_task(task, *, closed=("complete", TASK_SUPERSEDED_STATE)) -> bool:
    return str(task.get("state") or "") not in closed


def plan_recording(store: LabelingStore, recording_id: str) -> list[dict[str, object]]:
    """Describe, per keypoint lineage, what carrying forward would do."""

    tasks = store.list_tasks(recording_id=recording_id)
    checkpoints = store.list_recording_applied_checkpoints(
        recording_id=recording_id, workflow_kind="keypoints"
    )
    pending = store.list_recording_applied_checkpoints(
        recording_id=recording_id, workflow_kind="keypoints", states=("active", "applying")
    )
    edit_times = checkpoint_edit_times(
        checkpoints,
        store.list_recording_events(recording_id=recording_id, event_type="checkpoint_keypoints"),
    )
    plans = []
    for archive in sorted({task_archive(t) for t in tasks if task_run(t) and task_archive(t)}):
        root = zarr.open_group(str(archive), mode="r", use_consolidated=False)
        keypoints = read_family_lineage(root, KEYPOINT_FAMILY)
        masks = read_family_lineage(root, MASK_FAMILY)
        archive_keypoint_tasks = [
            t for t in tasks
            if t.get("workflow_kind") == "keypoints" and task_archive(t) == archive
        ]
        components = {keypoints.component(str(task_run(t))) for t in archive_keypoint_tasks}
        for component in sorted(components, key=sorted):
            if len(component) < 2:
                continue
            target = keypoints.newest_leaf(component)
            members = lineage_tasks(
                tasks, archive=archive, family=KEYPOINT_FAMILY, component=component
            )
            target_tasks = [t for t in members if task_run(t) == target]
            rows = stranded_keypoint_rows(
                root=root, checkpoints=checkpoints, lineage=keypoints, target_run=target,
                pending_checkpoints=pending, edit_times=edit_times,
            )
            plans.append(
                {
                    "recording_id": recording_id,
                    "archive": str(archive),
                    "target_run": target,
                    "target_task_ids": [str(t["task_id"]) for t in target_tasks],
                    "target_task_states": [str(t.get("state") or "") for t in target_tasks],
                    "older_keypoint_tasks": [
                        {
                            "task_id": str(t["task_id"]),
                            "run": task_run(t),
                            "state": str(t.get("state") or ""),
                            "unapplied_checkpoints": store.count_unapplied_session_checkpoints(
                                task_id=str(t["task_id"])
                            ),
                        }
                        for t in members
                        if task_run(t) != target
                    ],
                    "rows": [row.as_dict() for row in rows],
                }
            )
        for component in sorted(
            {masks.component(str(task_run(t))) for t in tasks
             if t.get("workflow_kind") == "subject_mask_component" and task_archive(t) == archive},
            key=sorted,
        ):
            if len(component) < 2:
                continue
            newest = masks.newest_leaf(component)
            members = lineage_tasks(tasks, archive=archive, family=MASK_FAMILY, component=component)
            plans.append(
                {
                    "recording_id": recording_id,
                    "archive": str(archive),
                    "mask_lineage_newest_run": newest,
                    "older_mask_tasks": [
                        {
                            "task_id": str(t["task_id"]),
                            "run": task_run(t),
                            "state": str(t.get("state") or ""),
                            "unapplied_checkpoints": store.count_unapplied_session_checkpoints(
                                task_id=str(t["task_id"])
                            ),
                        }
                        for t in members
                        if task_run(t) != newest
                    ],
                }
            )
    return plans


def _summary(plan: dict[str, object]) -> dict[str, object]:
    if "rows" not in plan:
        return {
            "mask_lineage_newest_run": plan["mask_lineage_newest_run"],
            "older_open_mask_tasks": sum(
                1 for t in plan["older_mask_tasks"] if _open_task(t)
            ),
        }
    counts: dict[str, int] = {}
    for row in plan["rows"]:
        counts[row["disposition"]] = counts.get(row["disposition"], 0) + 1
    return {
        "target_run": plan["target_run"],
        "target_task_ids": plan["target_task_ids"],
        "rows_by_disposition": counts,
        "manual_landmarks_to_carry": sum(
            len(row["manual_keypoints"]) for row in plan["rows"] if row["disposition"] == "carry"
        ),
        "older_tasks_with_unapplied_checkpoints": [
            t["task_id"] for t in plan["older_keypoint_tasks"] if t["unapplied_checkpoints"]
        ],
    }


def carry_rows(store: LabelingStore, plan: dict[str, object], *, user: str) -> dict[str, object]:
    """Stage and apply the plan's carryable rows in the newest version's task."""

    from fisheye.labeling.web import _refresh_registry_for_scope
    from fisheye.labeling.web_keypoint_checkpoint_apply import apply_keypoint_checkpoints
    from fisheye.labeling.web_keypoint_checkpoints import (
        keypoint_checkpoint_state,
        stage_keypoint_checkpoint,
    )
    from fisheye.labeling.web_runtimes import _get_keypoint_runtime
    from fisheye.tune import keypoint_review_backend as backend

    rows = [row for row in plan["rows"] if row["disposition"] == "carry"]
    if not rows:
        return {"carried_rows": 0}
    if len(plan["target_task_ids"]) != 1:
        raise RuntimeError(f"Expected one task on {plan['target_run']}, found {plan['target_task_ids']}")
    task_id = plan["target_task_ids"][0]
    task = store.get_task(task_id)
    if task is None or str(task.get("state") or "") not in LABELER_START_TASK_STATES:
        raise RuntimeError(f"Newest-version task {task_id} is not open for labeling")
    if store.count_unapplied_session_checkpoints(task_id=task_id):
        raise RuntimeError(f"Task {task_id} has unapplied checkpoints; apply them first")
    lease = store.create_session(task_id=task_id, user=user, client_label="carry_forward_tail_keypoints")
    session_id = str(getattr(lease, "session_id", None) or lease["session_id"])
    session = store.get_session(session_id)
    try:
        runtime = _get_keypoint_runtime(_RuntimeState(), session)
        positions = {int(roi): i for i, roi in enumerate(np.asarray(runtime.review_session.failures).tolist())}
        for row in rows:
            roi = int(row["roi_idx"])
            if roi not in positions:
                raise RuntimeError(f"Row {roi} is outside task {task_id}'s scope")
            runtime.position = positions[roi]
            stage_keypoint_checkpoint(
                store,
                runtime,
                user=user,
                operation="replace_points",
                points=row["points"],
                carried_from={
                    "schema": CARRY_SCHEMA,
                    "source_run": row["source_run"],
                    "source_task_id": row["source_task_id"],
                    "source_checkpoint_ids": row["checkpoint_ids"],
                    "source_edited_at_utc": row["applied_at_utc"],
                    "manual_keypoints": row["manual_keypoints"],
                },
            )
        state = keypoint_checkpoint_state(store, runtime)
        digest = str(state.get("checkpoint_snapshot_sha256") or "")
        apply_id = str(uuid.uuid4())
        result = apply_keypoint_checkpoints(
            store, runtime, backend, apply_id=apply_id, checkpoint_snapshot_sha256=digest
        )
        store.record_event(
            task_id=task_id,
            recording_id=str(task["recording_id"]),
            user=user,
            event_type="apply_keypoint_session_checkpoints",
            target={"apply_id": apply_id, "refined_run": plan["target_run"], "rows": result.get("rows")},
            after={**result, "carry_forward_schema": CARRY_SCHEMA},
        )
        if not _refresh_registry_for_scope(
            store=store,
            task_id=task_id,
            recording_id=str(task["recording_id"]),
            user=user,
            workflow_kind="keypoints",
            scope=task.get("scope") or {},
            zarr_path=str(runtime.review_session.zarr_path),
            dataset_id=str(task.get("dataset_id") or "") or None,
            zarr_use=str(task.get("zarr_use") or "") or None,
        ):
            raise RuntimeError("Registry refresh failed after carry-forward Apply")
        store.mark_session_checkpoint_apply_effects_complete(
            task_id=task_id, component_name="keypoints", apply_id=apply_id
        )
        return {"carried_rows": len(rows), "apply_id": apply_id, "task_id": task_id}
    finally:
        store.close_session(session_id=session_id, user=user)


def verify_carried(plan: dict[str, object]) -> dict[str, object]:
    """Re-read the newest version and confirm every carried landmark landed."""

    root = zarr.open_group(plan["archive"], mode="r", use_consolidated=False)
    group = root[f"{KEYPOINT_FAMILY}/{plan['target_run']}"]
    points = np.asarray(group["keypoints_roi"][:], dtype=np.float64)
    manual = np.asarray(group["keypoint_manual_edit"][:], dtype=bool)
    missing = []
    for row in plan["rows"]:
        if row["disposition"] != "carry":
            continue
        roi, idx = int(row["roi_idx"]), row["manual_keypoints"]
        expected = np.asarray(row["points"], dtype=np.float64)[idx]
        if not (np.allclose(points[roi][idx], expected) and manual[roi][idx].all()):
            missing.append(roi)
    return {"verified_rows": sum(1 for r in plan["rows"] if r["disposition"] == "carry") - len(missing),
            "rows_not_matching": missing}


def close_older_versions(store: LabelingStore, plans: list[dict[str, object]], *, user: str) -> dict[str, object]:
    """Supersede open older-version tasks without unapplied work; report the rest."""

    superseded, blocked = [], []
    for plan in plans:
        older = plan.get("older_keypoint_tasks") or plan.get("older_mask_tasks") or []
        for task in older:
            if not _open_task(task):
                continue
            if task["unapplied_checkpoints"]:
                blocked.append(task["task_id"])
                continue
            superseded.extend(
                store.supersede_tasks(
                    task_ids=[task["task_id"]],
                    user=user,
                    reason="replaced_by_tail_successor",
                    details={"carry_forward_schema": CARRY_SCHEMA},
                )
            )
    retired = []
    for recording_id in sorted({str(plan["recording_id"]) for plan in plans}):
        for task in store.list_tasks(recording_id=recording_id):
            if task.get("workflow_kind") == "subject_mask_component" and task.get("state") == "complete":
                retired.extend(
                    store.retire_untouched_review_successors(source_task_id=str(task["task_id"]), user=user)
                )
    return {"superseded": superseded, "retired_mask_successors": retired, "blocked_with_unapplied": blocked}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--user", required=True)
    parser.add_argument("--recording-id", action="append", default=[])
    parser.add_argument("--apply", action="store_true", help="write; the default is a dry run")
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args(argv)
    store = LabelingStore(args.store)
    store.initialize()
    recording_ids = args.recording_id or sorted(
        {str(t["recording_id"]) for t in store.list_tasks() if t.get("workflow_kind") == "keypoints"}
    )
    report: dict[str, object] = {"schema": CARRY_SCHEMA, "mode": "apply" if args.apply else "dry_run", "recordings": {}}
    all_plans = []
    for recording_id in recording_ids:
        plans = plan_recording(store, recording_id)
        all_plans.extend(plans)
        entry = {"plans": [_summary(p) for p in plans]}
        if args.apply:
            entry["carry"] = []
            for plan in plans:
                if "rows" in plan:
                    outcome = carry_rows(store, plan, user=args.user)
                    if outcome.get("carried_rows"):
                        outcome.update(verify_carried(plan))
                    entry["carry"].append(outcome)
        report["recordings"][recording_id] = entry
        print(json.dumps({recording_id: entry}, default=str), flush=True)
    if args.apply:
        report["close_older_versions"] = close_older_versions(store, all_plans, user=args.user)
        print(json.dumps(report["close_older_versions"]), flush=True)
    if args.report_json:
        args.report_json.write_text(json.dumps({**report, "detail": all_plans}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
