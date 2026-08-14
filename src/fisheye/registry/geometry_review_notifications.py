"""Idempotent one-shot notifications for actionable geometry-review states."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote, urlencode

from fisheye.labeling.notifications import (
    LabelingNotification,
    LabelingNotificationConfig,
    send_labeling_notification,
)

from .geometry_review import (
    GeometryReviewTransition,
    actionable_geometry_transitions,
    load_geometry_review_queue,
)


NOTIFICATION_STATE_SCHEMA = "palette.geometry_review_notification_state.v1"
NOTIFICATION_RESULT_SCHEMA = "palette.geometry_review_notification_scan.v1"


@dataclass(frozen=True)
class GeometryReviewNotificationScan:
    registry_path: Path
    state_db: Path
    observed: tuple[GeometryReviewTransition, ...]
    new: tuple[GeometryReviewTransition, ...]
    delivery: Mapping[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": NOTIFICATION_RESULT_SCHEMA,
            "registry_path": str(self.registry_path),
            "state_db": str(self.state_db),
            "observed_actionable_count": len(self.observed),
            "new_actionable_count": len(self.new),
            "new_event_keys": [item.event_key for item in self.new],
            "delivery": dict(self.delivery),
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_state_path(state_db: Path, *, registry_path: Path) -> Path:
    resolved = state_db.expanduser().resolve()
    registry = registry_path.expanduser().resolve()
    if resolved == registry:
        raise ValueError("Notification state must not be the canonical registry.")
    if any(parent.name.endswith(".zarr") for parent in (resolved, *resolved.parents)):
        raise ValueError("Notification state must be outside canonical analysis Zarrs.")
    return resolved


def _open_state(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS geometry_review_notification_events (
            event_key TEXT PRIMARY KEY,
            dataset_id TEXT NOT NULL,
            recording_id TEXT NOT NULL,
            zarr_path TEXT NOT NULL,
            stage TEXT NOT NULL,
            semantic_state TEXT NOT NULL,
            run_id TEXT NOT NULL,
            digest TEXT NOT NULL,
            first_seen_utc TEXT NOT NULL,
            last_seen_utc TEXT NOT NULL,
            delivered_utc TEXT,
            delivery_status TEXT,
            notification_id TEXT
        );
        CREATE TABLE IF NOT EXISTS geometry_review_notification_scans (
            scan_id INTEGER PRIMARY KEY AUTOINCREMENT,
            scanned_utc TEXT NOT NULL,
            registry_path TEXT NOT NULL,
            observed_count INTEGER NOT NULL,
            new_count INTEGER NOT NULL,
            result_json TEXT NOT NULL
        );
        """
    )
    return conn


def _record_observed(
    conn: sqlite3.Connection,
    transitions: Sequence[GeometryReviewTransition],
    *,
    now: str,
) -> set[str]:
    delivered = {
        str(row["event_key"])
        for row in conn.execute(
            "SELECT event_key FROM geometry_review_notification_events "
            "WHERE delivered_utc IS NOT NULL"
        ).fetchall()
    }
    conn.executemany(
        """
        INSERT INTO geometry_review_notification_events (
            event_key, dataset_id, recording_id, zarr_path, stage,
            semantic_state, run_id, digest, first_seen_utc, last_seen_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(event_key) DO UPDATE SET last_seen_utc=excluded.last_seen_utc
        """,
        [
            (
                item.event_key,
                item.dataset_id,
                item.recording_id,
                str(item.zarr_path),
                item.stage,
                item.semantic_state,
                item.run_id,
                item.digest,
                now,
                now,
            )
            for item in transitions
        ],
    )
    conn.commit()
    return delivered


def _event_url(item: GeometryReviewTransition, *, base_url: str | None) -> str | None:
    if not base_url:
        return None
    query = urlencode(
        {
            "dataset_id": item.dataset_id,
            "run_id": item.run_id,
        },
        quote_via=quote,
    )
    return f"{base_url.rstrip('/')}?{query}"


def build_geometry_review_digest(
    transitions: Sequence[GeometryReviewTransition],
    *,
    recipients: str,
    base_url: str | None,
) -> LabelingNotification:
    recordings = sorted({item.recording_id for item in transitions})
    lines = [
        "Palette registered-dish geometry review has new actionable states.",
        "",
        f"Recordings: {len(recordings)}",
        f"Transitions: {len(transitions)}",
        "",
    ]
    for item in transitions:
        lines.extend(
            [
                f"- {item.recording_id}",
                f"  dataset: {item.dataset_id}",
                f"  stage/state: {item.stage} / {item.semantic_state}",
                f"  exact run: {item.run_id}",
                f"  digest: {item.digest}",
            ]
        )
        link = _event_url(item, base_url=base_url)
        if link:
            lines.append(f"  review: {link}")
    lines.extend(
        [
            "",
            "This notification records an operational alert only. It does not "
            "select geometry, publish a candidate, apply a gate, or mark scientific work complete.",
        ]
    )
    return LabelingNotification(
        kind="geometry_review_digest",
        to_email=recipients.strip(),
        to_user="geometry-review-operators",
        subject=(
            f"Palette geometry review: {len(recordings)} recording"
            f"{'s' if len(recordings) != 1 else ''} need attention"
        ),
        text_body="\n".join(lines),
    )


def _no_delivery(status: str, *, reason: str, dry_run: bool = False) -> dict[str, Any]:
    return {
        "ok": status in {"no_new_events", "dry_run"},
        "status": status,
        "transport": "none",
        "reason": reason,
        "sent": False,
        "queued": False,
        "dry_run": dry_run,
    }


def scan_geometry_review_notifications(
    *,
    registry_path: str | Path,
    state_db: str | Path,
    recipients: str,
    config: LabelingNotificationConfig,
    dry_run: bool = False,
) -> GeometryReviewNotificationScan:
    """Scan once, send at most one digest, and durably deduplicate delivery."""

    registry = Path(registry_path).expanduser().resolve()
    state = _validate_state_path(Path(state_db), registry_path=registry)
    queue = load_geometry_review_queue(registry, include_inactive=False)
    observed = tuple(actionable_geometry_transitions(queue))
    now = _utc_now()
    with _open_state(state) as conn:
        delivered_keys = _record_observed(conn, observed, now=now)
        new = tuple(item for item in observed if item.event_key not in delivered_keys)
        if not new:
            delivery: Mapping[str, Any] = _no_delivery(
                "no_new_events", reason="all actionable transitions were already delivered"
            )
        elif dry_run:
            delivery = {
                **_no_delivery(
                    "dry_run",
                    reason="digest rendered without transport or dedup completion",
                    dry_run=True,
                ),
                "subject": build_geometry_review_digest(
                    new, recipients=recipients, base_url=config.base_url
                ).subject,
            }
        else:
            notification = build_geometry_review_digest(
                new, recipients=recipients, base_url=config.base_url
            )
            delivery = send_labeling_notification(
                notification,
                actor_user="geometry-review-scanner",
                config=config,
                context={
                    "event_count": len(new),
                    "recording_count": len({item.recording_id for item in new}),
                    "event_keys": [item.event_key for item in new],
                },
            )
        if bool(delivery.get("ok")) and delivery.get("status") in {"queued", "sent"}:
            conn.executemany(
                """
                UPDATE geometry_review_notification_events
                SET delivered_utc=?, delivery_status=?, notification_id=?
                WHERE event_key=?
                """,
                [
                    (
                        now,
                        str(delivery.get("status")),
                        str(delivery.get("notification_id") or ""),
                        item.event_key,
                    )
                    for item in new
                ],
            )
        scan_payload = {
            "schema": NOTIFICATION_RESULT_SCHEMA,
            "delivery": dict(delivery),
            "new_event_keys": [item.event_key for item in new],
        }
        conn.execute(
            """
            INSERT INTO geometry_review_notification_scans (
                scanned_utc, registry_path, observed_count, new_count, result_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                now,
                str(registry),
                len(observed),
                len(new),
                json.dumps(scan_payload, sort_keys=True),
            ),
        )
        conn.commit()
    return GeometryReviewNotificationScan(
        registry_path=registry,
        state_db=state,
        observed=observed,
        new=new,
        delivery=delivery,
    )


__all__ = [
    "GeometryReviewNotificationScan",
    "build_geometry_review_digest",
    "scan_geometry_review_notifications",
]
