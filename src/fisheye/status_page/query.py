"""Registry access helpers for the status page server."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import sqlite3
from typing import Any, Iterable, Optional, Sequence

from ..registry.db import RegistryPaths
from ..registry.stage_catalog import recording_status_stage_ids
from .models import HealthReport

REQUIRED_STATUS_OBJECTS: tuple[str, ...] = (
    "recording_step_status_latest",
    "recording_step_status_wide",
    "recording_step_status_history",
)

WIDE_SEARCH_COLUMNS: tuple[str, ...] = (
    "Recording",
    "Camera",
    "Use",
    "Purpose",
    "Detect",
    "Detect Quality",
    "Refine Detect",
    "Detect Group",
    "Detect Review",
    "Crop",
    "Crop Review",
    "Keypoints",
    "Refined Keypoints (analysis/train)",
    "Keypoint Review",
    "Eye Masks",
    "Refined Eye Masks",
    "Subject Masks",
    "Refined Subject Masks",
    "Eye Mask Review",
    "Arena Assignment",
    "Track",
    "Stimulus",
    "Calib",
    "Tuning",
)

WIDE_BLOCKING_COLUMNS: tuple[str, ...] = (
    "Zarr",
    "Import",
    "BG Full",
    "BG DS",
    "Detect",
    "Detect Quality",
    "Refine Detect",
    "Crop",
    "Keypoints",
    "Refined Keypoints (analysis/train)",
    "Eye Masks",
    "Refined Eye Masks",
    "Subject Masks",
    "Refined Subject Masks",
    "Arena Assignment",
    "Track",
    "Stimulus",
    "Calib",
    "Tuning",
)

STEP_SORT_ORDER: tuple[str, ...] = recording_status_stage_ids()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_json_text(value: Any) -> Any:
    text = _normalize_text(value)
    if text is None:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def _coerce_int(
    value: Any,
    *,
    default: int,
    minimum: int,
    maximum: int,
    field: str,
) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer.") from exc
    if parsed < minimum:
        parsed = minimum
    if parsed > maximum:
        parsed = maximum
    return parsed


def _registry_object_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type IN ('table', 'view') AND name = ?
        LIMIT 1;
        """,
        (name,),
    ).fetchone()
    return row is not None


def _object_columns(conn: sqlite3.Connection, object_name: str) -> tuple[str, ...]:
    if not _registry_object_exists(conn, object_name):
        return tuple()
    rows = conn.execute(f"PRAGMA table_info({_quote_ident(object_name)});").fetchall()
    return tuple(str(row["name"]) for row in rows if row["name"] is not None)


def _as_dict_rows(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def _blocking_expr(available_columns: set[str]) -> str:
    cols = [col for col in WIDE_BLOCKING_COLUMNS if col in available_columns]
    if not cols:
        return "0"
    clauses: list[str] = []
    for col in cols:
        quoted = _quote_ident(col)
        clauses.append(
            "("
            f"UPPER(COALESCE(CAST({quoted} AS TEXT), '')) LIKE 'MISS%'"
            f" OR UPPER(COALESCE(CAST({quoted} AS TEXT), '')) LIKE 'ERR%'"
            f" OR UPPER(COALESCE(CAST({quoted} AS TEXT), '')) LIKE 'FAIL%'"
            ")"
        )
    return " OR ".join(clauses)


def _decorate_status_row(row: dict[str, Any]) -> dict[str, Any]:
    for key in ("review_status_json", "details_json"):
        parsed_key = key.removesuffix("_json")
        row[parsed_key] = _parse_json_text(row.get(key))
    return row


def _build_wide_dataset_lookup(
    conn: sqlite3.Connection,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
            COALESCE(NULLIF(trim(recording_id), ''), dataset_id) AS recording_key,
            COALESCE(camera_id, 'unknown') AS camera_key,
            lower(trim(COALESCE(CAST(zarr_use AS TEXT), ''))) AS use_key,
            COUNT(DISTINCT dataset_id) AS dataset_count,
            GROUP_CONCAT(DISTINCT dataset_id) AS dataset_ids_csv
        FROM recording_step_status_latest
        GROUP BY
            COALESCE(NULLIF(trim(recording_id), ''), dataset_id),
            COALESCE(camera_id, 'unknown'),
            lower(trim(COALESCE(CAST(zarr_use AS TEXT), '')));
        """
    ).fetchall()
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        recording_key = _normalize_text(row["recording_key"]) or ""
        camera_key = _normalize_text(row["camera_key"]) or "unknown"
        use_key = (_normalize_text(row["use_key"]) or "").lower()
        dataset_ids_csv = _normalize_text(row["dataset_ids_csv"]) or ""
        dataset_ids = [part for part in dataset_ids_csv.split(",") if part]
        dataset_count = int(row["dataset_count"] or len(dataset_ids) or 0)
        dataset_id = dataset_ids[0] if dataset_count == 1 and dataset_ids else None
        out[(recording_key, camera_key, use_key)] = {
            "dataset_count": dataset_count,
            "dataset_ids": dataset_ids,
            "dataset_id": dataset_id,
        }
    return out


def _attach_wide_dataset_identity(
    rows: list[dict[str, Any]],
    *,
    lookup: dict[tuple[str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        recording_key = _normalize_text(enriched.get("Recording")) or ""
        camera_key = _normalize_text(enriched.get("Camera")) or "unknown"
        use_key = (_normalize_text(enriched.get("Use")) or "").lower()
        match = lookup.get((recording_key, camera_key, use_key))
        if match is None and use_key == "—":
            # Some rows may render use as em-dash in wide view while latest holds empty text.
            match = lookup.get((recording_key, camera_key, ""))
        if match is None:
            enriched["_dataset_id"] = None
            enriched["_dataset_count"] = 0
            enriched["_dataset_ids"] = []
        else:
            enriched["_dataset_id"] = match["dataset_id"]
            enriched["_dataset_count"] = int(match["dataset_count"])
            enriched["_dataset_ids"] = list(match["dataset_ids"])
        out.append(enriched)
    return out


def resolve_registry_path(path: Optional[Path], *, cwd: Optional[Path] = None) -> Path:
    if path is not None:
        return path.expanduser().resolve()
    root = (cwd or Path.cwd()).expanduser().resolve()
    return RegistryPaths.from_env(root).path.expanduser().resolve()


def validate_registry_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Registry not found: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"Registry path is not a file: {resolved}")
    return resolved


def open_readonly_connection(path: Path) -> sqlite3.Connection:
    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


def _open_status_connection(path: Path) -> sqlite3.Connection:
    resolved = validate_registry_path(path)
    conn = open_readonly_connection(resolved)
    missing = [name for name in REQUIRED_STATUS_OBJECTS if not _registry_object_exists(conn, name)]
    if missing:
        conn.close()
        missing_text = ",".join(sorted(missing))
        raise RuntimeError(f"missing_required_objects:{missing_text}")
    return conn


def build_health_report(
    registry_path: Path,
    *,
    required_objects: Sequence[str] = REQUIRED_STATUS_OBJECTS,
) -> HealthReport:
    checked_utc = _utc_now()
    resolved = registry_path.expanduser().resolve()
    details: dict[str, object] = {
        "sqlite_ok": False,
        "required_objects": {},
    }
    ok = True
    message = "ok"

    conn: Optional[sqlite3.Connection] = None
    try:
        validate_registry_path(resolved)
        conn = open_readonly_connection(resolved)
        conn.execute("SELECT 1;").fetchone()
        details["sqlite_ok"] = True

        required_map: dict[str, bool] = {}
        missing: list[str] = []
        for name in required_objects:
            present = _registry_object_exists(conn, name)
            required_map[name] = present
            if not present:
                missing.append(name)
        details["required_objects"] = required_map
        if missing:
            ok = False
            message = f"missing_required_objects:{','.join(sorted(missing))}"
    except Exception as exc:
        ok = False
        message = str(exc)
    finally:
        if conn is not None:
            conn.close()

    return HealthReport(
        ok=ok,
        registry_path=str(resolved),
        checked_utc=checked_utc,
        message=message,
        details=details,
    )


def assert_registry_ready(
    registry_path: Path,
    *,
    required_objects: Iterable[str] = REQUIRED_STATUS_OBJECTS,
) -> Path:
    resolved = validate_registry_path(registry_path)
    report = build_health_report(resolved, required_objects=tuple(required_objects))
    if report.ok:
        return resolved
    raise RuntimeError(
        f"Registry is not status-page ready: {report.message} "
        f"(path={report.registry_path})"
    )


def query_status_summary(registry_path: Path) -> dict[str, Any]:
    conn = _open_status_connection(registry_path)
    try:
        latest = conn.execute(
            """
            SELECT
                COUNT(*) AS status_rows_total,
                COALESCE(SUM(CASE WHEN lower(status) = 'missing' THEN 1 ELSE 0 END), 0) AS status_rows_missing,
                COALESCE(SUM(CASE WHEN lower(status) = 'error' THEN 1 ELSE 0 END), 0) AS status_rows_error,
                COUNT(DISTINCT dataset_id) AS datasets_total,
                COUNT(DISTINCT CASE
                    WHEN recording_id IS NOT NULL AND trim(recording_id) <> '' THEN recording_id
                    ELSE NULL
                END) AS recordings_total,
                MAX(updated_utc) AS latest_updated_utc
            FROM recording_step_status_latest;
            """
        ).fetchone()

        wide_columns = set(_object_columns(conn, "recording_step_status_wide"))
        blocking_expr = _blocking_expr(wide_columns)
        wide = conn.execute(
            f"""
            SELECT
                COUNT(*) AS wide_rows_total,
                COALESCE(SUM(CASE WHEN ({blocking_expr}) THEN 1 ELSE 0 END), 0) AS wide_rows_blocking
            FROM recording_step_status_wide;
            """
        ).fetchone()

        use_counts: dict[str, int] = {}
        if "Use" in wide_columns:
            rows = conn.execute(
                """
                SELECT lower(trim(COALESCE(CAST("Use" AS TEXT), ''))) AS use_name, COUNT(*) AS use_count
                FROM recording_step_status_wide
                GROUP BY lower(trim(COALESCE(CAST("Use" AS TEXT), '')));
                """
            ).fetchall()
            for row in rows:
                name = _normalize_text(row["use_name"]) or "unknown"
                use_counts[name] = int(row["use_count"] or 0)

        return {
            "generated_utc": _utc_now(),
            "datasets_total": int(latest["datasets_total"] or 0),
            "recordings_total": int(latest["recordings_total"] or 0),
            "status_rows_total": int(latest["status_rows_total"] or 0),
            "status_rows_missing": int(latest["status_rows_missing"] or 0),
            "status_rows_error": int(latest["status_rows_error"] or 0),
            "wide_rows_total": int(wide["wide_rows_total"] or 0),
            "wide_rows_blocking": int(wide["wide_rows_blocking"] or 0),
            "latest_updated_utc": _normalize_text(latest["latest_updated_utc"]),
            "wide_use_counts": use_counts,
        }
    finally:
        conn.close()


def query_status_heartbeat(registry_path: Path) -> dict[str, Any]:
    conn = _open_status_connection(registry_path)
    try:
        latest = conn.execute(
            """
            SELECT
                MAX(updated_utc) AS latest_updated_utc,
                COUNT(*) AS status_rows_total
            FROM recording_step_status_latest;
            """
        ).fetchone()
        wide = conn.execute(
            """
            SELECT COUNT(*) AS wide_rows_total
            FROM recording_step_status_wide;
            """
        ).fetchone()
        return {
            "generated_utc": _utc_now(),
            "latest_updated_utc": _normalize_text(latest["latest_updated_utc"]),
            "status_rows_total": int(latest["status_rows_total"] or 0),
            "wide_rows_total": int(wide["wide_rows_total"] or 0),
        }
    finally:
        conn.close()


def query_status_wide(
    registry_path: Path,
    *,
    q: Optional[str] = None,
    zarr_use: Optional[str] = None,
    only_blocking: bool = False,
    limit: int = 200,
    offset: int = 0,
) -> dict[str, Any]:
    safe_limit = _coerce_int(limit, default=200, minimum=1, maximum=5000, field="limit")
    safe_offset = _coerce_int(offset, default=0, minimum=0, maximum=5_000_000, field="offset")
    search_text = _normalize_text(q)
    use_filter = _normalize_text(zarr_use)

    conn = _open_status_connection(registry_path)
    try:
        wide_columns = _object_columns(conn, "recording_step_status_wide")
        wide_colset = set(wide_columns)

        where_clauses: list[str] = []
        where_params: list[Any] = []

        if use_filter is not None:
            if "Use" not in wide_colset:
                raise RuntimeError("recording_step_status_wide is missing required column: Use")
            where_clauses.append("lower(trim(COALESCE(CAST(\"Use\" AS TEXT), ''))) = ?")
            where_params.append(use_filter.lower())

        if search_text is not None:
            search_columns = [col for col in WIDE_SEARCH_COLUMNS if col in wide_colset]
            if not search_columns:
                search_columns = list(wide_columns)
            if search_columns:
                search_exprs = [
                    f"lower(COALESCE(CAST({_quote_ident(col)} AS TEXT), '')) LIKE ?"
                    for col in search_columns
                ]
                where_clauses.append("(" + " OR ".join(search_exprs) + ")")
                needle = f"%{search_text.lower()}%"
                where_params.extend([needle] * len(search_exprs))

        if only_blocking:
            blocking_expr = _blocking_expr(wide_colset)
            where_clauses.append(f"({blocking_expr})")

        where_sql = ""
        if where_clauses:
            where_sql = " WHERE " + " AND ".join(where_clauses)

        order_columns = [col for col in ("Recording", "Camera", "Use") if col in wide_colset]
        if order_columns:
            order_sql = ", ".join(f"{_quote_ident(col)} ASC" for col in order_columns)
        else:
            order_sql = "1 ASC"

        total_row = conn.execute(
            f"SELECT COUNT(*) AS total_rows FROM recording_step_status_wide{where_sql};",
            tuple(where_params),
        ).fetchone()
        total_rows = int(total_row["total_rows"] or 0)

        rows = conn.execute(
            (
                "SELECT * FROM recording_step_status_wide"
                f"{where_sql}"
                f" ORDER BY {order_sql}"
                " LIMIT ? OFFSET ?;"
            ),
            tuple(where_params + [safe_limit, safe_offset]),
        ).fetchall()
        payload_rows = _as_dict_rows(rows)
        if {"Recording", "Camera", "Use"} <= wide_colset:
            lookup = _build_wide_dataset_lookup(conn)
            payload_rows = _attach_wide_dataset_identity(payload_rows, lookup=lookup)

        return {
            "rows": payload_rows,
            "columns": list(wide_columns),
            "hidden_columns": ["_dataset_id", "_dataset_count", "_dataset_ids"],
            "total_rows": total_rows,
            "limit": safe_limit,
            "offset": safe_offset,
            "returned_rows": len(rows),
            "filters": {
                "q": search_text,
                "zarr_use": use_filter.lower() if use_filter else None,
                "only_blocking": bool(only_blocking),
            },
            "generated_utc": _utc_now(),
        }
    finally:
        conn.close()


def query_dataset_status(
    registry_path: Path,
    *,
    dataset_id: str,
) -> dict[str, Any]:
    dataset_text = _normalize_text(dataset_id)
    if dataset_text is None:
        raise ValueError("dataset_id is required.")

    conn = _open_status_connection(registry_path)
    try:
        case_fragments = [
            f"WHEN '{step_name}' THEN {index}"
            for index, step_name in enumerate(STEP_SORT_ORDER, start=1)
        ]
        step_sort_sql = "CASE lower(step_name) " + " ".join(case_fragments) + " ELSE 999 END"
        rows = conn.execute(
            f"""
            SELECT *
            FROM recording_step_status_latest
            WHERE dataset_id = ?
            ORDER BY {step_sort_sql}, lower(step_name), updated_utc DESC;
            """,
            (dataset_text,),
        ).fetchall()

        payload_rows = [_decorate_status_row(dict(row)) for row in rows]
        return {
            "dataset_id": dataset_text,
            "rows": payload_rows,
            "row_count": len(payload_rows),
            "generated_utc": _utc_now(),
        }
    finally:
        conn.close()


def query_status_history(
    registry_path: Path,
    *,
    dataset_id: str,
    step_name: Optional[str] = None,
    limit: int = 200,
) -> dict[str, Any]:
    dataset_text = _normalize_text(dataset_id)
    if dataset_text is None:
        raise ValueError("dataset_id is required.")
    step_text = _normalize_text(step_name)
    safe_limit = _coerce_int(limit, default=200, minimum=1, maximum=5000, field="limit")

    conn = _open_status_connection(registry_path)
    try:
        where_parts = ["dataset_id = ?"]
        params: list[Any] = [dataset_text]
        if step_text is not None:
            where_parts.append("lower(step_name) = ?")
            params.append(step_text.lower())
        where_sql = " AND ".join(where_parts)

        rows = conn.execute(
            f"""
            SELECT *
            FROM recording_step_status_history
            WHERE {where_sql}
            ORDER BY recorded_utc DESC, event_id DESC
            LIMIT ?;
            """,
            tuple(params + [safe_limit]),
        ).fetchall()
        payload_rows = [_decorate_status_row(dict(row)) for row in rows]
        return {
            "dataset_id": dataset_text,
            "step_name": step_text.lower() if step_text else None,
            "rows": payload_rows,
            "row_count": len(payload_rows),
            "limit": safe_limit,
            "generated_utc": _utc_now(),
        }
    finally:
        conn.close()
