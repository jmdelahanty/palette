"""Registry writer for acquisition video stream encoder rows (migration 079)."""

from __future__ import annotations

from typing import Any, Dict, Iterable

from .extractors.acquisition_video_stream_encoders import ENCODER_COLUMNS


class RegistryAcquisitionVideoStreamEncodersMixin:
    def replace_acquisition_video_stream_encoders(
        self, dataset_id: str, rows: Iterable[Dict[str, Any]]
    ) -> None:
        """Replace a dataset's stream encoder rows whole."""

        if not self._sqlite_object_exists("acquisition_video_stream_encoders", object_types=("table",)):
            self._migration_079_acquisition_video_stream_encoders()
        with self._maybe_transaction():
            self.conn.execute(
                "DELETE FROM acquisition_video_stream_encoders WHERE dataset_id = ?;",
                (str(dataset_id),),
            )
            for row in rows:
                payload = {name: row.get(name) for name in ENCODER_COLUMNS}
                payload["dataset_id"] = str(dataset_id)
                self.conn.execute(
                    f"INSERT INTO acquisition_video_stream_encoders ({', '.join(ENCODER_COLUMNS)}) "
                    f"VALUES ({', '.join(':' + name for name in ENCODER_COLUMNS)});",
                    payload,
                )


__all__ = ["RegistryAcquisitionVideoStreamEncodersMixin"]
