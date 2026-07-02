from __future__ import annotations

from typing import Any, Mapping, Sequence

from fisheye.registry.db import Registry


def _table_columns(registry: Registry, table_name: str) -> set[str]:
    rows = registry.conn.execute(f"PRAGMA table_info({table_name});").fetchall()
    return {str(row["name"]) for row in rows}


def _insert_row(
    registry: Registry,
    *,
    table_name: str,
    conflict_columns: Sequence[str],
    values: Mapping[str, Any],
) -> None:
    table_columns = _table_columns(registry, table_name)
    payload = {key: value for key, value in dict(values).items() if key in table_columns}
    if "updated_utc" in table_columns:
        payload.setdefault("updated_utc", "2026-02-01T00:00:00+00:00")
    if "quality_updated_utc" in table_columns:
        payload.setdefault("quality_updated_utc", "2026-02-01T00:00:00+00:00")
    missing_conflict = [column for column in conflict_columns if column not in payload]
    if missing_conflict:
        raise AssertionError(f"missing conflict column(s) for {table_name}: {missing_conflict}")

    columns = list(payload)
    placeholders = ", ".join(f":{column}" for column in columns)
    conflict = ", ".join(conflict_columns)
    update_columns = [column for column in columns if column not in conflict_columns]
    if update_columns:
        update_sql = " DO UPDATE SET " + ", ".join(
            f"{column}=excluded.{column}" for column in update_columns
        )
    else:
        update_sql = " DO NOTHING"

    registry.conn.execute(
        f"""
        INSERT INTO {table_name} ({", ".join(columns)})
        VALUES ({placeholders})
        ON CONFLICT({conflict}){update_sql};
        """,
        payload,
    )
    registry.conn.commit()


def insert_eye_mask_performance(registry: Registry, **values: Any) -> None:
    _insert_row(
        registry,
        table_name="eye_mask_performance",
        conflict_columns=("dataset_id", "stage_group", "run_name"),
        values=values,
    )


def insert_eye_mask_data_profile(registry: Registry, **values: Any) -> None:
    _insert_row(
        registry,
        table_name="eye_mask_data_profile",
        conflict_columns=("dataset_id", "profile_run"),
        values=values,
    )


def replace_eye_mask_data_profile(
    registry: Registry,
    dataset_id: str,
    records: Sequence[Mapping[str, Any]],
) -> None:
    registry.conn.execute(
        "DELETE FROM eye_mask_data_profile WHERE dataset_id = ?;",
        (str(dataset_id),),
    )
    for record in records:
        payload = dict(record)
        payload["dataset_id"] = str(dataset_id)
        insert_eye_mask_data_profile(registry, **payload)
