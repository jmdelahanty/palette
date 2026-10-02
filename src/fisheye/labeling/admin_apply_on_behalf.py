"""Operator command: Apply a labeler's saved mask edits on their behalf.

Labelers sometimes save rows (checkpoints) but never press Apply, so the
edits never reach the training archive. This command applies one task's
saved subject-mask edits through the exact route the mask editor uses: it
runs a private in-process server on the store, opens a session as the task's
assignee and makes the same /subject-mask/apply request the editor's Apply
button makes. Every ownership check, write lock, QC step and tail refresh is
therefore the editor's own; nothing here writes labels.

Dry run by default. With --execute it refuses when the labeler has the task
open or an Apply is unfinished, takes a validated store backup, applies, and
records an ``admin_apply_on_behalf`` event naming the operator; the applied
checkpoints keep the labeler as their author.

    scripts/py -m fisheye.labeling.admin_apply_on_behalf \\
        --store ~/.palette/labeling_work.sqlite --task-id TASK --actor OPERATOR [--execute]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import urllib.error
import urllib.request
import uuid
from collections.abc import Sequence
from http.server import ThreadingHTTPServer
from pathlib import Path

from fisheye.labeling.assignment_store import LabelingStore
from fisheye.labeling.store_backup import DEFAULT_BACKUP_DIR, backup_labeling_store

ADMIN_APPLY_CLIENT_LABEL = "admin_apply_on_behalf"
ADMIN_APPLY_EVENT = "admin_apply_on_behalf"
SUPPORTED_WORKFLOWS = {"subject_mask_component"}


class AdminApplyRefused(RuntimeError):
    """The task is not in a state where an operator may apply for the labeler."""


def apply_plan(store: LabelingStore, task_id: str, *, retry_apply_id: str | None = None) -> dict[str, object]:
    """Read-only summary of what an Apply on behalf would do, with refusals.

    With ``retry_apply_id`` the plan is to resume that Apply, whose pixels were
    written but whose derived effects (QC) are still owed, as the editor's own
    retry does; it must name this task's unfinished receipt.
    """

    task = store.get_task(task_id)
    if task is None:
        raise AdminApplyRefused(f"Unknown task_id: {task_id}")
    conn = store.conn
    count = lambda sql: int(conn.execute(sql, (task_id,)).fetchone()[0])  # noqa: E731
    open_sessions = [
        dict(row) for row in conn.execute(
            """
            SELECT session_id, user, created_at_utc, last_seen_at_utc, client_label
            FROM labeling_sessions
            WHERE task_id = ? AND closed_at_utc IS NULL
              AND expires_at_utc > strftime('%Y-%m-%dT%H:%M:%fZ', 'now');
            """,
            (task_id,),
        )
    ]
    pending_rows = count(
        "SELECT COUNT(*) FROM labeling_session_checkpoints WHERE task_id = ? AND state = 'active';"
    )
    applying_rows = count(
        "SELECT COUNT(*) FROM labeling_session_checkpoints WHERE task_id = ? AND state = 'applying';"
    )
    unfinished_receipts = count(
        """
        SELECT COUNT(*) FROM labeling_checkpoint_apply_receipts
        WHERE task_id = ? AND (state != 'applied' OR secondary_effects_state != 'complete');
        """
    )
    pending_by_user = {
        str(row[0]): int(row[1])
        for row in conn.execute(
            """
            SELECT user, COUNT(*) FROM labeling_session_checkpoints
            WHERE task_id = ? AND state = 'active' GROUP BY user;
            """,
            (task_id,),
        )
    }
    workflow = str(task.get("workflow_kind") or "")
    assignee = str(task.get("assignee_user") or "")
    refusals = []
    if workflow not in SUPPORTED_WORKFLOWS:
        refusals.append(f"workflow {workflow!r} is not supported (subject_mask_component only)")
    if str(task.get("assignment_status") or "") != "active" or not assignee:
        refusals.append("the recording has no active assignee")
    if str(task.get("state") or "") in {"complete", "superseded"}:
        refusals.append(f"task state is {task.get('state')!r}")
    if open_sessions:
        refusals.append(f"{len(open_sessions)} open editor session(s) on this task; the labeler may be working")
    unfinished_ids = [
        str(row[0]) for row in conn.execute(
            """
            SELECT apply_id FROM labeling_checkpoint_apply_receipts
            WHERE task_id = ? AND (state != 'applied' OR secondary_effects_state != 'complete');
            """,
            (task_id,),
        )
    ]
    if retry_apply_id:
        if unfinished_ids != [retry_apply_id]:
            refusals.append(f"retry needs exactly this task's unfinished Apply {retry_apply_id}; found {unfinished_ids}")
        if applying_rows:
            refusals.append("rows are still being claimed by an Apply; let it finish")
    else:
        if applying_rows or unfinished_receipts:
            refusals.append(
                f"an Apply is unfinished on this task ({unfinished_ids}); retry it with --retry-apply-id"
            )
        if not pending_rows:
            refusals.append("no saved, unapplied rows")
    return {
        "task_id": task_id,
        "recording_id": str(task.get("recording_id") or ""),
        "workflow_kind": workflow,
        "component_name": str(task.get("component_name") or ""),
        "task_state": str(task.get("state") or ""),
        "assignee": assignee,
        "pending_rows": pending_rows,
        "pending_rows_by_user": pending_by_user,
        "open_sessions": open_sessions,
        "unfinished_receipts": unfinished_receipts,
        "unfinished_apply_ids": unfinished_ids,
        "retry_apply_id": retry_apply_id,
        "refusals": refusals,
        "ok": not refusals,
    }


def _request(base: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
    request = urllib.request.Request(
        base + path,
        data=None if payload is None else json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def apply_on_behalf(
    store_path: Path,
    task_id: str,
    *,
    actor: str,
    backup_dir: Path | None = None,
    skip_backup: bool = False,
    retry_apply_id: str | None = None,
) -> dict[str, object]:
    """Apply one task's saved edits for its assignee through the editor's route."""

    from fisheye.labeling import web as labeling_web

    store = LabelingStore(store_path)
    try:
        plan = apply_plan(store, task_id, retry_apply_id=retry_apply_id)
        if not plan["ok"]:
            raise AdminApplyRefused("; ".join(plan["refusals"]))
        assignee = str(plan["assignee"])
        backup = None if skip_backup else backup_labeling_store(
            store_path, backup_dir or DEFAULT_BACKUP_DIR, label=Path(store_path).stem
        )
        config = labeling_web.ServerConfig(
            store_path=Path(store_path),
            host="127.0.0.1",
            port=0,
            fixed_user=assignee,
            auth_header="X-Forwarded-User",
            session_ttl_seconds=3600,
            admin_users=(actor,),
        )
        state = labeling_web.ServerState(store=store, config=config)
        server = ThreadingHTTPServer(("127.0.0.1", 0), labeling_web._make_handler(state))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        lease = store.create_session(task_id=task_id, user=assignee, client_label=ADMIN_APPLY_CLIENT_LABEL)
        session_id = str(lease.session_id)
        apply_id = retry_apply_id or str(uuid.uuid4())
        try:
            host, port = server.server_address
            base = f"http://{host}:{port}/api/sessions/{session_id}/subject-mask"
            status, current = _request(base, "/state")
            if status != 200 or not current.get("ok"):
                raise RuntimeError(f"Could not load the task state ({status}): {current.get('error')}: {current.get('details')}")
            target_token = (current.get("state") or current).get("target_token")
            status, response = _request(base, "/apply", {"apply_id": apply_id, "target_token": target_token})
        finally:
            store.close_session(session_id=session_id, user=assignee)
            server.shutdown()
            server.server_close()
            thread.join(timeout=10)
        result = response.get("result") if isinstance(response.get("result"), dict) else {}
        outcome = {
            "ok": status == 200 and bool(response.get("ok")),
            "http_status": status,
            "apply_id": apply_id,
            "session_id": session_id,
            "applied_checkpoint_count": result.get("applied_checkpoint_count"),
            "edit_revision_before": result.get("edit_revision_before"),
            "edit_revision_after": result.get("edit_revision_after"),
            "qc_status": result.get("qc_status"),
            "already_applied": result.get("already_applied"),
            "retry": bool(retry_apply_id),
            "error": response.get("error"),
            "details": response.get("details"),
        }
        store.record_event(
            task_id=task_id,
            recording_id=str(plan["recording_id"]),
            user=actor,
            event_type=ADMIN_APPLY_EVENT,
            target={"apply_id": apply_id, "labeler": assignee, "session_id": session_id},
            before={"pending_rows": plan["pending_rows"], "backup_path": (backup or {}).get("backup_path")},
            after=outcome,
        )
        return {"plan": plan, "backup": backup, "outcome": outcome}
    finally:
        store.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--actor", required=True, help="Operator recorded on the admin_apply_on_behalf event.")
    parser.add_argument("--execute", action="store_true", help="Apply; without it, only print the plan.")
    parser.add_argument("--retry-apply-id", help="Resume this task's written Apply whose QC/effects are still owed.")
    parser.add_argument("--backup-dir", type=Path, default=Path(os.environ.get("PALETTE_LABELING_BACKUP_DIR", DEFAULT_BACKUP_DIR)))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    store_path = args.store.expanduser().resolve()
    if not args.execute:
        store = LabelingStore(store_path)
        try:
            plan = apply_plan(store, args.task_id, retry_apply_id=args.retry_apply_id)
        finally:
            store.close()
        print(json.dumps({"dry_run": True, "plan": plan}, indent=2, default=str))
        return 0 if plan["ok"] else 2
    try:
        report = apply_on_behalf(
            store_path, args.task_id, actor=args.actor, backup_dir=args.backup_dir,
            retry_apply_id=args.retry_apply_id,
        )
    except AdminApplyRefused as exc:
        print(json.dumps({"ok": False, "refused": str(exc)}, indent=2))
        return 2
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["outcome"]["ok"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
