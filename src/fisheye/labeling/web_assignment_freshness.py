"""Assignment snapshot and handoff freshness helpers for web labeling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .assignment_store import LabelingStore


def _handoff_dataset_queue_state_counts_impl(handoffs: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for handoff in handoffs:
        code = str(handoff.get("dataset_queue_state_code") or "")
        if not code:
            state = handoff.get("dataset_queue_state") if isinstance(handoff.get("dataset_queue_state"), Mapping) else {}
            code = str(state.get("code") or "")
        if code:
            counts[code] = counts.get(code, 0) + 1
    return dict(sorted(counts.items()))


def _handoff_assignment_snapshot_from_work_impl(work: Mapping[str, object], user: str) -> dict[str, object]:
    rows_by_recording: dict[str, dict[str, object]] = {}
    recordings = work.get("recordings")
    if isinstance(recordings, list):
        for recording in recordings:
            if not isinstance(recording, Mapping):
                continue
            recording_id = str(recording.get("recording_id") or "").strip()
            if not recording_id:
                continue
            assignee_user = str(recording.get("assignee_user") or user).strip()
            assignment_status = str(recording.get("assignment_status") or "active").strip() or "active"
            rows_by_recording[recording_id] = {
                "recording_id": recording_id,
                "assignee_user": assignee_user,
                "status": assignment_status,
            }
    rows = [rows_by_recording[key] for key in sorted(rows_by_recording)]
    return {
        "schema": "palette.web_labeling_assignment_snapshot.v1",
        "user": str(user),
        "recording_count": len(rows),
        "recording_ids": [str(row["recording_id"]) for row in rows],
        "assignments": rows,
    }


def _assignment_snapshot_rows_impl(snapshot: Mapping[str, object], fallback_user: str) -> list[dict[str, object]]:
    raw_rows = snapshot.get("assignments")
    rows: list[dict[str, object]] = []
    if isinstance(raw_rows, list):
        for row in raw_rows:
            if not isinstance(row, Mapping):
                continue
            recording_id = str(row.get("recording_id") or "").strip()
            if not recording_id:
                continue
            rows.append(
                {
                    "recording_id": recording_id,
                    "assignee_user": str(row.get("assignee_user") or fallback_user).strip(),
                    "status": str(row.get("status") or "active").strip() or "active",
                }
            )
        return rows
    recording_ids = snapshot.get("recording_ids")
    if isinstance(recording_ids, list):
        for recording_id_value in recording_ids:
            recording_id = str(recording_id_value or "").strip()
            if not recording_id:
                continue
            rows.append(
                {
                    "recording_id": recording_id,
                    "assignee_user": str(snapshot.get("user") or fallback_user).strip(),
                    "status": "active",
                }
            )
    return rows


def _inspect_handoff_assignment_freshness_impl(
    manifests: Sequence[Mapping[str, object]],
    *,
    store: LabelingStore | None,
    package_kind: str = "",
) -> dict[str, object]:
    if store is None:
        return {
            "checked_against_current_store": False,
            "ok": None,
            "status": "not_checked",
            "expected_recording_count": 0,
            "matched_recording_count": 0,
            "stale_recording_count": 0,
            "extra_current_assignment_count": 0,
            "snapshot_missing_count": 0,
            "snapshot_missing_users": [],
            "stale_recordings": [],
            "extra_current_assignments": [],
        }
    current_assignments = store.list_assignments(status=None)
    current_by_recording = {
        str(row.get("recording_id") or ""): row
        for row in current_assignments
        if str(row.get("recording_id") or "")
    }
    expected_count = 0
    matched_count = 0
    stale_recordings: list[dict[str, object]] = []
    extra_current_assignments: list[dict[str, object]] = []
    snapshot_missing_users: list[str] = []
    expected_recording_ids: set[str] = set()
    expected_recording_ids_by_user: dict[str, set[str]] = {}
    snapshot_users: set[str] = set()
    for manifest in manifests:
        user = str(manifest.get("user") or "").strip()
        snapshot = manifest.get("assignment_snapshot")
        if not isinstance(snapshot, Mapping):
            snapshot_missing_users.append(user or str(manifest.get("output_dir") or "unknown"))
            continue
        snapshot_users.add(user)
        expected_rows = _assignment_snapshot_rows(snapshot, user)
        expected_count += len(expected_rows)
        for expected in expected_rows:
            recording_id = str(expected.get("recording_id") or "")
            expected_user = str(expected.get("assignee_user") or user)
            expected_status = str(expected.get("status") or "active")
            expected_recording_ids.add(recording_id)
            expected_recording_ids_by_user.setdefault(expected_user, set()).add(recording_id)
            current = current_by_recording.get(recording_id)
            current_user = str((current or {}).get("assignee_user") or "")
            current_status = str((current or {}).get("status") or "")
            if current is not None and current_user == expected_user and current_status == expected_status:
                matched_count += 1
                continue
            stale_recordings.append(
                {
                    "recording_id": recording_id,
                    "expected_user": expected_user,
                    "expected_status": expected_status,
                    "current_user": current_user or None,
                    "current_status": current_status or None,
                }
            )
    multi_user_package = str(package_kind).startswith(("batch", "launch"))
    for assignment in current_assignments:
        recording_id = str(assignment.get("recording_id") or "").strip()
        current_user = str(assignment.get("assignee_user") or "").strip()
        current_status = str(assignment.get("status") or "").strip()
        if not recording_id or current_status != "active":
            continue
        if multi_user_package:
            if recording_id in expected_recording_ids:
                continue
        else:
            if current_user not in snapshot_users:
                continue
            if recording_id in expected_recording_ids_by_user.get(current_user, set()):
                continue
        extra_current_assignments.append(
            {
                "recording_id": recording_id,
                "current_user": current_user,
                "current_status": current_status,
            }
        )
    extra_current_assignments = sorted(
        extra_current_assignments,
        key=lambda row: (str(row.get("current_user") or ""), str(row.get("recording_id") or "")),
    )
    ok = not snapshot_missing_users and not stale_recordings and not extra_current_assignments
    return {
        "checked_against_current_store": True,
        "ok": ok,
        "status": "current" if ok else ("unverified" if snapshot_missing_users else "stale"),
        "expected_recording_count": expected_count,
        "matched_recording_count": matched_count,
        "stale_recording_count": len(stale_recordings),
        "extra_current_assignment_count": len(extra_current_assignments),
        "snapshot_missing_count": len(snapshot_missing_users),
        "snapshot_missing_users": sorted(set(snapshot_missing_users)),
        "stale_recordings": stale_recordings,
        "extra_current_assignments": extra_current_assignments,
    }


def _assignment_snapshot_from_assignments(
    assignments: Sequence[Mapping[str, object]],
    *,
    user: str | None = None,
) -> dict[str, object]:
    rows_by_recording: dict[str, dict[str, object]] = {}
    for assignment in assignments:
        recording_id = str(assignment.get("recording_id") or "").strip()
        if not recording_id:
            continue
        rows_by_recording[recording_id] = {
            "recording_id": recording_id,
            "assignee_user": str(assignment.get("assignee_user") or "").strip(),
            "status": str(assignment.get("status") or "active").strip() or "active",
        }
    rows = [rows_by_recording[key] for key in sorted(rows_by_recording)]
    snapshot: dict[str, object] = {
        "schema": "palette.web_labeling_assignment_snapshot.v1",
        "recording_count": len(rows),
        "recording_ids": [str(row["recording_id"]) for row in rows],
        "assignments": rows,
    }
    if user is not None:
        snapshot["user"] = str(user)
    return snapshot


# Preserve original helper names inside this module so moved helpers can call each other.
_handoff_dataset_queue_state_counts = _handoff_dataset_queue_state_counts_impl
_handoff_assignment_snapshot_from_work = _handoff_assignment_snapshot_from_work_impl
_assignment_snapshot_rows = _assignment_snapshot_rows_impl
_inspect_handoff_assignment_freshness = _inspect_handoff_assignment_freshness_impl
