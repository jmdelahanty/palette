"""Registry writer for derived recording orientation status (migration 078)."""

from __future__ import annotations

from typing import Any, Dict

from fisheye.shared.batch_logging import utc_now as _utc_now


class RegistryOrientationStatusMixin:
    _ORIENTATION_STATUS_COLUMNS = (
        "recording_id", "session_uuid", "citrus_session_uuid", "h5_relative_path", "rig_id", "arena_id",
        "camera_serial", "orientation_status", "verified_link", "basis",
        "evidence_sha256", "reason", "camera_to_dish_parity", "status_schema_id",
        "status_schema_version", "status_json", "updated_utc",
    )

    def replace_recording_orientation_status(
        self, recording_id: str, row: Dict[str, Any]
    ) -> None:
        """Replace a recording's orientation-status row whole."""

        if not self._sqlite_object_exists("recording_orientation_status", object_types=("table",)):
            self._migration_078_recording_orientation_status()
        columns = self._ORIENTATION_STATUS_COLUMNS
        payload = {name: row.get(name) for name in columns}
        payload["recording_id"] = str(recording_id)
        payload["updated_utc"] = payload["updated_utc"] or _utc_now()
        with self._maybe_transaction():
            self.conn.execute(
                f"INSERT OR REPLACE INTO recording_orientation_status ({', '.join(columns)}) "
                f"VALUES ({', '.join(':' + name for name in columns)});",
                payload,
            )


__all__ = ["RegistryOrientationStatusMixin"]
