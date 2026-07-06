"""Assignment manifest validation helpers for web labeling control-plane workflows."""

from __future__ import annotations

from typing import Mapping, Sequence

from .admin_dashboard import _known_labeler_status
from .assignment_store import LabelingStore


def _active_assignee_user_issues(
    store: LabelingStore,
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    issues: list[dict[str, object]] = []
    for row in rows:
        assignee_user = str(row.get("assignee_user") or row.get("user") or "").strip()
        if not assignee_user:
            continue
        status = _known_labeler_status(store, assignee_user)
        if bool(status.get("is_active_labeling_user")):
            continue
        issues.append(
            {
                "code": "inactive_or_unknown_assignee_user",
                "assignee_user": assignee_user,
                "recording_id": str(row.get("recording_id") or ""),
                **({"source_line": row["_source_line"]} if row.get("_source_line") is not None else {}),
                "assignee_user_status": status,
                "details": (
                    "Assignments can only be created or updated for users with an active "
                    "row in the labeling_users SQLite table. Add or activate the user first."
                ),
            }
        )
    return issues
