"""Background worker that drains owed subject-mask Apply effects.

Enabled by ``serve --background-apply-effects``.  The Apply request returns
after the pixel commit and SQLite finalization; this single daemon thread then
runs the same effects sequence the inline path runs
(:func:`run_apply_effects_locked`).

* **Source of truth is SQLite.**  The queue is the set of applied receipts with
  ``secondary_effects_state='pending'``; the in-memory event only wakes the
  thread.  On start the worker scans for pending receipts, so work owed before
  a restart or crash resumes.
* **One Apply at a time, in Apply order.**  Receipts are processed in
  ``applied_at_utc`` order under the refined-run write lock.  When the oldest
  owed receipt of a mask run cannot run now, later receipts of that run wait.
* **Failures are durable and classified.**  Each attempt is recorded as a
  ``subject_mask_apply_effects_attempt`` task event.  Transient failures
  (I/O, lock contention, registry pending, waiting on paired keypoint Apply)
  retry with backoff; everything else is a refusal that is not retried until a
  person retries that ``apply_id``.
"""

from __future__ import annotations

import sqlite3
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

from .web_subject_mask_apply_effects import (
    ATTEMPT_EVENT,
    WORKER_USER,
    latest_effects_attempt,
    registry_scope_from_row,
    reopen_mask_run,
    run_apply_effects_locked,
)
from .web_subject_mask_apply_state import pending_mask_run_effects, require_mask_apply_ownership

RETRY_BACKOFF_SECONDS = (30.0, 120.0, 600.0, 3600.0)
IDLE_POLL_SECONDS = 60.0
TRANSIENT_RUNTIME_MARKERS = (
    "registry refresh remains pending",
    "Finish the pending mask Apply",
    "Apply the paired keypoint task",
)
BACKGROUND_EFFECTS_EVENT = "apply_subject_mask_session_checkpoints_background_effects"


class ApplyEffectsRefused(RuntimeError):
    """The owed effects cannot be reconstructed from durable state."""


def classify_failure(exc: BaseException) -> str:
    """``transient`` (retry with backoff) or ``refused`` (needs a person)."""

    if isinstance(exc, ApplyEffectsRefused):
        return "refused"
    if isinstance(exc, (OSError, sqlite3.OperationalError)):
        return "transient"
    if isinstance(exc, RuntimeError) and any(marker in str(exc) for marker in TRANSIENT_RUNTIME_MARKERS):
        return "transient"
    return "refused"


def _utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _parse_utc(value: object) -> float | None:
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except (TypeError, ValueError):
        return None


def build_effects_runtime(store, receipt):
    """Rebuild a subject-mask runtime for one receipt from durable store rows.

    The task row supplies scope (archive, component, registry scope); the
    receipt's applied checkpoints pin the refined run and the labeler.  The
    runtime is built by the same builder browser sessions use, with a scratch
    session cache so live sessions are not touched.
    """

    from .web_runtimes import _get_subject_mask_runtime

    task_id, apply_id = str(receipt["task_id"]), str(receipt["apply_id"])
    task = store.get_task(task_id)
    if task is None or str(task.get("workflow_kind") or "") != "subject_mask_component":
        raise ApplyEffectsRefused(f"Apply {apply_id} task {task_id} is not a subject-mask task.")
    checkpoints = store.get_applied_session_checkpoints_by_apply_id(task_id=task_id, apply_id=apply_id)
    if not checkpoints:
        raise ApplyEffectsRefused(f"Apply {apply_id} has no applied checkpoints.")
    target_paths = {str(row.get("target_run_path") or "") for row in checkpoints}
    target_path = next(iter(target_paths))
    prefix = "refined_subject_masks_runs/"
    if len(target_paths) != 1 or not target_path.startswith(prefix):
        raise ApplyEffectsRefused(f"Apply {apply_id} checkpoints do not bind one refined subject-mask run.")
    refined_run = target_path[len(prefix):]
    scope = dict(task.get("scope") or {})
    if str(scope.get("refined_run") or refined_run) != refined_run:
        raise ApplyEffectsRefused(f"Apply {apply_id} run differs from its task scope.")
    scope["refined_run"] = refined_run  # never let the builder select or create a run
    user = str(checkpoints[0].get("user") or "")
    session = {
        **task,
        "scope": scope,
        "session_id": f"apply-effects:{apply_id}",
        "user": user,
    }
    runtime = _get_subject_mask_runtime(SimpleNamespace(subject_mask_sessions={}), session)
    if runtime.refined.run_name != refined_run or runtime.component_name != str(receipt["component_name"]):
        raise ApplyEffectsRefused(f"Apply {apply_id} rebuilt runtime does not match its receipt.")
    return runtime, user, registry_scope_from_row(task)


class ApplyEffectsWorker:
    def __init__(
        self,
        store,
        *,
        refresh_registry: Callable[..., bool],
        on_complete: Callable[[str, str], None] | None = None,
        notify: Callable[[dict, dict], None] | None = None,
        backoff_seconds: tuple[float, ...] = RETRY_BACKOFF_SECONDS,
        lock_timeout_seconds: float = 30.0,
        idle_poll_seconds: float = IDLE_POLL_SECONDS,
    ) -> None:
        self.store = store
        self.refresh_registry = refresh_registry
        self.on_complete = on_complete
        self.notify = notify
        self.backoff_seconds = tuple(backoff_seconds)
        self.lock_timeout_seconds = float(lock_timeout_seconds)
        self.idle_poll_seconds = float(idle_poll_seconds)
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._progress = threading.Condition()
        self._force_retry = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="palette-apply-effects", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 30.0) -> None:
        """Take no new work; let the current Apply finish (bounded)."""

        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def wake(self, *, force_retry: bool = False) -> None:
        if force_retry:
            self._force_retry = True
        self._wake.set()

    def wait_for_progress(self, timeout: float) -> None:
        with self._progress:
            self._progress.wait(timeout=max(0.0, float(timeout)))

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.clear()
            try:
                delay = self.run_once()
            except Exception as exc:  # keep the thread alive; receipts stay pending
                print(f"apply_effects_worker_error={exc!r}", file=sys.stderr, flush=True)
                delay = self.idle_poll_seconds
            self._wake.wait(timeout=max(0.05, delay))

    def run_once(self) -> float:
        """Process every due receipt once; return seconds until the next is due."""

        force, self._force_retry = self._force_retry, False
        blocked_runs: set[tuple[str, str]] = set()
        next_due = self.idle_poll_seconds
        for receipt in self.store.list_all_pending_subject_mask_apply_effects():
            if self._stop.is_set():
                break
            path = str(receipt["zarr_path"])
            run_key = (str(Path(path).expanduser().resolve()) if path else "", str(receipt["refined_run"]))
            if not path:
                run_key = ("task", str(receipt["task_id"]))
            if run_key in blocked_runs:
                continue
            last = latest_effects_attempt(self.store, task_id=str(receipt["task_id"]), apply_id=str(receipt["apply_id"]))
            if last.get("status") == "refused":
                blocked_runs.add(run_key)
                continue
            if last.get("status") == "failed" and not force:
                due_at = _parse_utc(last.get("next_attempt_at_utc")) or 0.0
                if due_at > time.time():
                    blocked_runs.add(run_key)
                    next_due = min(next_due, due_at - time.time())
                    continue
            if not self.process(receipt, previous_attempt=int(last.get("attempt") or 0)):
                blocked_runs.add(run_key)
                retry = latest_effects_attempt(self.store, task_id=str(receipt["task_id"]), apply_id=str(receipt["apply_id"]))
                due_at = _parse_utc(retry.get("next_attempt_at_utc"))
                if due_at is not None:
                    next_due = min(next_due, due_at - time.time())
            with self._progress:
                self._progress.notify_all()
        return next_due

    def _record_attempt(self, receipt, **after) -> None:
        self.store.record_event(
            task_id=str(receipt["task_id"]),
            recording_id=str(receipt.get("recording_id") or ""),
            user=WORKER_USER,
            event_type=ATTEMPT_EVENT,
            target={"apply_id": str(receipt["apply_id"])},
            after=after,
        )

    def _finish_effects(self, runtime, *, user, apply_id, derived, registry_scope) -> None:
        """Record the effects and write any deferred review status before the
        receipt is marked complete, so a failed status write keeps effects owed."""

        from .web_subject_mask_deferred_review import apply_pending_deferred_review_locked

        self.store.record_event(
            task_id=runtime.task_id, recording_id=runtime.recording_id, user=user,
            event_type=BACKGROUND_EFFECTS_EVENT, target={"apply_id": apply_id},
            after={"apply_id": apply_id, **derived},
        )
        reopen_mask_run(runtime)
        apply_pending_deferred_review_locked(
            self.store, runtime, refresh_registry=self.refresh_registry, registry_scope=registry_scope,
        )

    def process(self, receipt, *, previous_attempt: int = 0) -> bool:
        from fisheye.tune import refined_subject_mask_review as review_mod

        apply_id = str(receipt["apply_id"])
        attempt = int(previous_attempt) + 1
        self._record_attempt(receipt, status="running", attempt=attempt, started_at_utc=_utc(time.time()))
        try:
            runtime, user, registry_scope = build_effects_runtime(self.store, receipt)
            with review_mod._refined_subject_write_lock(
                runtime.zarr_path, refined_run=runtime.refined.run_name,
                timeout_seconds=self.lock_timeout_seconds,
            ):
                owed = [row for row in pending_mask_run_effects(self.store, runtime) if row["apply_id"] == apply_id]
                if owed:
                    require_mask_apply_ownership(self.store, runtime, apply_id)
                    run_apply_effects_locked(
                        store=self.store, runtime=runtime, root=reopen_mask_run(runtime),
                        apply_id=apply_id, expected_revision=int(owed[0]["edit_revision_after"]),
                        refresh_registry=self.refresh_registry, registry_scope=registry_scope, user=user,
                        before_complete=lambda derived: self._finish_effects(
                            runtime, user=user, apply_id=apply_id, derived=derived, registry_scope=registry_scope,
                        ),
                    )
        except Exception as exc:
            kind = classify_failure(exc)
            reason = (str(exc) or type(exc).__name__)[:1000]
            after = {"status": "failed" if kind == "transient" else "refused", "attempt": attempt,
                     "failure_kind": kind, "reason": reason, "error_type": type(exc).__name__}
            if kind == "transient":
                delay = self.backoff_seconds[min(attempt, len(self.backoff_seconds)) - 1]
                after["next_attempt_at_utc"] = _utc(time.time() + delay)
            self._record_attempt(receipt, **after)
            if self.notify is not None and (kind == "refused" or attempt == 1):
                try:
                    self.notify(dict(receipt), after)
                except Exception as notify_exc:
                    print(f"apply_effects_notify_error={notify_exc!r}", file=sys.stderr, flush=True)
            return False
        self._record_attempt(receipt, status="complete", attempt=attempt, completed_at_utc=_utc(time.time()))
        if self.on_complete is not None:
            try:
                self.on_complete(runtime.zarr_path, runtime.refined.run_name)
            except Exception:
                pass
        return True


def wait_for_prior_apply_effects(state, runtime, apply_id: str, *, timeout_seconds: float):
    """Wait (bounded) for other owed Applies on this run; None when clear.

    Returns an error payload when the prior Apply was refused or is still
    running at the deadline.  Waiting keeps one tail successor per Apply: the
    tail refresh requires the mask to still be at that Apply's revision.
    """

    from .web_subject_mask_apply_effects import apply_effects_status

    store = state.store
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    woken = None
    while True:
        others = [row for row in pending_mask_run_effects(store, runtime) if row["apply_id"] != apply_id]
        if not others:
            return None
        status = apply_effects_status(store, runtime) or {}
        if status.get("state") == "failed":
            return {
                "error": "previous_update_failed",
                "details": "The previous update of this mask failed and needs attention: " + str(status.get("reason") or ""),
                "apply_effects_status": status,
            }
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return {
                "error": "previous_update_still_running",
                "details": "The previous update of this mask is still running. Try Apply again shortly.",
                "apply_effects_status": status,
            }
        worker = state.apply_effects_worker
        if worker is not None and worker is not woken:
            worker.wake(force_retry=True)  # a waiting Apply is a retry trigger
            woken = worker
        if worker is not None:
            worker.wait_for_progress(min(remaining, 1.0))
        else:
            time.sleep(min(remaining, 0.25))


def refresh_live_runtimes(sessions, zarr_path: str, refined_run: str) -> None:
    """Reopen live browser runtimes of a run so they see worker-written attrs."""

    from fisheye.tune import refined_subject_mask_review as review_mod

    archive = Path(zarr_path).expanduser().resolve()
    for runtime in list(sessions.values()):
        try:
            if runtime.refined.run_name != refined_run or Path(runtime.zarr_path).expanduser().resolve() != archive:
                continue
            root = review_mod.open_zarr_root(runtime.zarr_path, mode="a")
            runtime.refined = review_mod._open_existing_refined_subject_run(root, refined_run)
            runtime.root = root
        except Exception:
            continue


def notify_admins_of_failure(store, admin_users, receipt, after) -> None:
    """Email configured admins (existing notification path) and audit the result."""

    from .notification_events import _notification_event_type, _notification_exception_result
    from .notifications import LabelingNotification, LabelingNotificationConfig, send_labeling_notification

    task_id, apply_id = str(receipt["task_id"]), str(receipt["apply_id"])
    recording_id = str(receipt.get("recording_id") or "")
    status = str(after.get("status") or "failed")
    retry_line = (
        f"It will retry automatically at {after.get('next_attempt_at_utc')}."
        if status == "failed" else "It will not retry automatically; retry that Apply or investigate."
    )
    body = "\n".join([
        "A background subject-mask Apply update (QC, tail versions, registry) did not finish.",
        "",
        f"Task: {task_id}",
        f"Recording: {recording_id}",
        f"Apply ID: {apply_id}",
        f"Attempt: {after.get('attempt')}",
        f"Reason: {after.get('reason')}",
        "",
        retry_line,
        "The applied mask pixels are saved; derived QC and tail versions stay marked stale until this finishes.",
    ])
    recipients = [str(user) for user in admin_users if str(user).strip()]
    if not recipients:
        store.record_event(
            task_id=task_id, recording_id=recording_id, user=WORKER_USER,
            event_type="apply_effects_failure_notification_skipped",
            target={"apply_id": apply_id}, after={"reason": "no_admin_users", "attempt": after.get("attempt")},
        )
        return
    config = LabelingNotificationConfig.from_env()
    for admin in recipients:
        user_row = store.get_labeling_user(admin) or {}
        notification = LabelingNotification(
            kind="apply_effects_failed",
            to_email=str(user_row.get("email") or ""),
            to_user=admin,
            subject=f"Palette labeling: background update {'refused' if status == 'refused' else 'failed'} ({task_id})",
            text_body=body,
        )
        try:
            result = send_labeling_notification(
                notification, actor_user=WORKER_USER, config=config,
                context={"task_id": task_id, "apply_id": apply_id, "status": status},
            )
        except Exception as exc:
            result = _notification_exception_result(kind="apply_effects_failed", to_user=admin, exc=exc)
        store.record_event(
            task_id=task_id, recording_id=recording_id, user=WORKER_USER,
            event_type=_notification_event_type(result, prefix="apply_effects_failure_notification"),
            target={"apply_id": apply_id, "admin_user": admin}, after=result,
        )


def start_worker_for_state(state, *, refresh_registry: Callable[..., bool]) -> ApplyEffectsWorker:
    """Create, attach, and start the server's single effects worker."""

    worker = ApplyEffectsWorker(
        state.store,
        refresh_registry=refresh_registry,
        on_complete=lambda zarr_path, run: refresh_live_runtimes(state.subject_mask_sessions, zarr_path, run),
        notify=lambda receipt, after: notify_admins_of_failure(
            state.store, state.config.admin_users, receipt, after,
        ),
    )
    state.apply_effects_worker = worker
    worker.start()
    return worker
