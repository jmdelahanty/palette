"""Shared writer API for recording step status rows."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .db import Registry

RECORDING_STEP_STATUSES = ("ok", "missing", "absent", "na", "error")
_RECORDING_STEP_STATUS_SET = frozenset(RECORDING_STEP_STATUSES)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _require_text(value: Any, *, field: str) -> str:
    text = _clean_text(value)
    if text is None:
        raise ValueError(f"{field} is required.")
    return text


def _normalize_float(value: Any, *, field: str) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a float or null.") from exc


def _normalize_int(value: Any, *, field: str) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer or null.") from exc


def _canonical_json_text(value: Any, *, field: str) -> Optional[str]:
    if value is None:
        return None

    payload = value
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8", "ignore")

    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            return None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{field} must be valid JSON text.") from exc

    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except TypeError as exc:
        raise ValueError(f"{field} must be JSON-serializable.") from exc


def normalize_recording_step_status(status: str) -> str:
    """Normalize and validate a recording step status."""
    normalized = _require_text(status, field="status").lower()
    if normalized not in _RECORDING_STEP_STATUS_SET:
        allowed = ", ".join(RECORDING_STEP_STATUSES)
        raise ValueError(f"status must be one of: {allowed}.")
    return normalized


def _resolve_recording_id(
    registry: Registry,
    *,
    dataset_id: str,
    recording_id: Optional[str],
) -> Optional[str]:
    explicit = _clean_text(recording_id)
    if explicit is not None:
        return explicit
    row = registry.conn.execute(
        """
        SELECT recording_id
        FROM datasets
        WHERE dataset_id = ?
        LIMIT 1;
        """,
        (dataset_id,),
    ).fetchone()
    if row is None:
        return None
    return _clean_text(row["recording_id"])


def upsert_recording_step_status(
    registry: Registry,
    *,
    dataset_id: str,
    step_name: str,
    status: str,
    run_name: Optional[str] = None,
    method: Optional[str] = None,
    coverage_pct: Optional[float] = None,
    review_status_json: Any = None,
    details_json: Any = None,
    updated_utc: Optional[str] = None,
    source: Optional[str] = None,
    recording_id: Optional[str] = None,
    zarr_mtime_ns: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Upsert one latest row and append one history event for a recording step status.

    The latest table is keyed by ``(dataset_id, step_name)`` and is idempotent.
    The history table appends a new event row for every successful call.
    """
    dataset_id_text = _require_text(dataset_id, field="dataset_id")
    step_name_text = _require_text(step_name, field="step_name")
    status_text = normalize_recording_step_status(status)

    payload: Dict[str, Any] = {
        "dataset_id": dataset_id_text,
        "recording_id": _resolve_recording_id(
            registry,
            dataset_id=dataset_id_text,
            recording_id=recording_id,
        ),
        "step_name": step_name_text,
        "status": status_text,
        "run_name": _clean_text(run_name),
        "method": _clean_text(method),
        "coverage_pct": _normalize_float(coverage_pct, field="coverage_pct"),
        "review_status_json": _canonical_json_text(
            review_status_json,
            field="review_status_json",
        ),
        "details_json": _canonical_json_text(
            details_json,
            field="details_json",
        ),
        "source": _clean_text(source),
        "zarr_mtime_ns": _normalize_int(zarr_mtime_ns, field="zarr_mtime_ns"),
        "updated_utc": _clean_text(updated_utc) or _utc_now(),
    }

    history_payload = dict(payload)
    history_payload["recorded_utc"] = _utc_now()

    with registry.conn:
        registry.conn.execute(
            """
            INSERT INTO recording_step_status (
                dataset_id,
                recording_id,
                step_name,
                status,
                run_name,
                method,
                coverage_pct,
                review_status_json,
                details_json,
                source,
                zarr_mtime_ns,
                updated_utc
            )
            VALUES (
                :dataset_id,
                :recording_id,
                :step_name,
                :status,
                :run_name,
                :method,
                :coverage_pct,
                :review_status_json,
                :details_json,
                :source,
                :zarr_mtime_ns,
                :updated_utc
            )
            ON CONFLICT(dataset_id, step_name) DO UPDATE SET
                recording_id=COALESCE(excluded.recording_id, recording_step_status.recording_id),
                status=excluded.status,
                run_name=excluded.run_name,
                method=excluded.method,
                coverage_pct=excluded.coverage_pct,
                review_status_json=excluded.review_status_json,
                details_json=excluded.details_json,
                source=excluded.source,
                zarr_mtime_ns=excluded.zarr_mtime_ns,
                updated_utc=excluded.updated_utc;
            """,
            payload,
        )

        registry.conn.execute(
            """
            INSERT INTO recording_step_status_history (
                dataset_id,
                recording_id,
                step_name,
                status,
                run_name,
                method,
                coverage_pct,
                review_status_json,
                details_json,
                source,
                zarr_mtime_ns,
                updated_utc,
                recorded_utc
            )
            VALUES (
                :dataset_id,
                :recording_id,
                :step_name,
                :status,
                :run_name,
                :method,
                :coverage_pct,
                :review_status_json,
                :details_json,
                :source,
                :zarr_mtime_ns,
                :updated_utc,
                :recorded_utc
            );
            """,
            history_payload,
        )

    return payload


__all__ = [
    "RECORDING_STEP_STATUSES",
    "normalize_recording_step_status",
    "upsert_recording_step_status",
]
