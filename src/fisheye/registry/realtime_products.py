"""Registry writer for Orange realtime products (migration 076)."""

from __future__ import annotations

from typing import Any, Dict, Iterable

from fisheye.shared.batch_logging import utc_now as _utc_now


class RegistryRealtimeProductsMixin:
    _REALTIME_PRODUCT_COLUMNS = (
        "dataset_id", "product", "recording_id", "camera_id", "declared", "status",
        "line_schema_id", "line_schema_version", "row_count", "header_rows",
        "result_rows", "no_result_rows", "failed_rows", "other_rows",
        "first_recording_frame_id", "last_recording_frame_id", "validation",
        "line_validator", "events_path", "events_sha256", "model_id", "model_schema",
        "engine_sha256", "engine_bytes", "weights_sha256", "onnx_sha256",
        "engine_manifest_sha256", "engine_manifest_run_id", "engine_manifest_status",
        "engine_precision", "engine_build_id", "record_sha256", "updated_utc",
    )

    def replace_recording_realtime_products(
        self, dataset_id: str, records: Iterable[Dict[str, Any]]
    ) -> None:
        """Mirror a dataset's realtime-products record exactly (rows replaced whole)."""

        if not self._sqlite_object_exists("recording_realtime_products", object_types=("table",)):
            self._migration_076_recording_realtime_products()
        columns = self._REALTIME_PRODUCT_COLUMNS
        with self._maybe_transaction():
            self.conn.execute(
                "DELETE FROM recording_realtime_products WHERE dataset_id = ?;",
                (str(dataset_id),),
            )
            for record in records:
                payload = {name: record.get(name) for name in columns}
                payload["dataset_id"] = str(dataset_id)
                payload["updated_utc"] = payload["updated_utc"] or _utc_now()
                self.conn.execute(
                    f"INSERT INTO recording_realtime_products ({', '.join(columns)}) "
                    f"VALUES ({', '.join(':' + name for name in columns)});",
                    payload,
                )



__all__ = ["RegistryRealtimeProductsMixin"]
