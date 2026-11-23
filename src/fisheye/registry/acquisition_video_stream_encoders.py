"""Registry writer for acquisition video stream encoder rows (migration 079)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from .extractors.acquisition_video_stream_encoders import (
    ENCODER_COLUMNS,
    extract_acquisition_video_stream_encoder_rows,
)


class RegistryAcquisitionVideoStreamEncodersMixin:
    def project_acquisition_video_stream_encoders(
        self, dataset_id: str, zarr_path: Any, streams: Iterable[Mapping[str, Any]]
    ) -> None:
        """Re-read every stream's recorder summary and replace the dataset's rows."""

        zarr_path = Path(zarr_path)
        recording_dir = zarr_path.parent.parent if zarr_path.parent.name == "zarr" else zarr_path.parent
        self.replace_acquisition_video_stream_encoders(
            dataset_id,
            extract_acquisition_video_stream_encoder_rows(recording_dir, streams, dataset_id=dataset_id),
        )

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
