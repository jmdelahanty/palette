"""Read-only conversion of historical detections to canonical benchmark input."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import zarr

from fisheye.shared.instance_keys import mint_detection_instance_keys
from fisheye.shared.zarr.benchmark_runtime import sha256_array, sha256_file
from fisheye.shared.zarr.detection_schema import (
    CANONICAL_DETECTION_SCHEMA_V1,
    CanonicalDetectionDimensions,
    derive_canonical_detection_geometry,
)


@dataclass(frozen=True)
class CanonicalDetectionBenchmarkInput:
    """Validated canonical arrays held in memory before timed writes."""

    dimensions: CanonicalDetectionDimensions
    arrays: Mapping[str, np.ndarray]
    source_identity: Mapping[str, object]

    def __post_init__(self) -> None:
        expected = CANONICAL_DETECTION_SCHEMA_V1.binding_paths
        if tuple(self.arrays) != expected:
            raise ValueError(
                "Benchmark input arrays must match canonical binding order exactly."
            )
        CANONICAL_DETECTION_SCHEMA_V1.require(
            self.arrays,
            dimensions=self.dimensions,
        )

    def as_manifest(self) -> dict[str, object]:
        return {
            "schema_id": "palette.canonical_detection_benchmark_input",
            "schema_version": 1,
            "dimensions": self.dimensions.as_manifest(),
            "source_identity": dict(self.source_identity),
            "canonical_arrays": {
                path: {
                    "shape": list(values.shape),
                    "dtype": str(values.dtype),
                    "sha256": sha256_array(values),
                }
                for path, values in self.arrays.items()
            },
        }


def build_canonical_detection_benchmark_input(
    source_arrays: Mapping[str, Any],
    *,
    recording_identity: str,
    frame_count: int,
    source_width: int,
    source_height: int,
    frame_limit: int | None = None,
    source_identity: Mapping[str, object] | None = None,
) -> CanonicalDetectionBenchmarkInput:
    """Convert one legacy/current detect table to exact canonical v1 arrays."""

    total_frames = int(frame_count)
    if total_frames < 0:
        raise ValueError("frame_count cannot be negative.")
    selected_frames = total_frames if frame_limit is None else int(frame_limit)
    if selected_frames < 0 or selected_frames > total_frames:
        raise ValueError("frame_limit must be within the source frame domain.")

    source_frame_indices = np.asarray(source_arrays["frame_indices"][:])
    source_bbox = np.asarray(source_arrays["bbox_norm_coords"][:])
    source_scores = np.asarray(source_arrays["scores"][:])
    source_class_ids = np.asarray(source_arrays["class_ids"][:])
    row_count = int(source_frame_indices.shape[0])
    if not (
        source_bbox.shape == (row_count, 4)
        and source_scores.shape == (row_count,)
        and source_class_ids.shape == (row_count,)
    ):
        raise ValueError("Source detection arrays do not share one row cardinality.")

    source_frames_i64 = np.asarray(source_frame_indices, dtype=np.int64)
    if source_frames_i64.size > 1 and np.any(np.diff(source_frames_i64) < 0):
        raise ValueError("Source detection rows must already be frame sorted.")
    stop = int(np.searchsorted(source_frames_i64, selected_frames, side="left"))
    frame_indices = np.asarray(source_frame_indices[:stop], dtype=np.int32)
    bbox_norm = np.asarray(source_bbox[:stop], dtype=np.float32)
    scores = np.asarray(source_scores[:stop], dtype=np.float32)
    class_ids = np.asarray(source_class_ids[:stop], dtype=np.int32)
    source_acquisition_frames = frame_indices.astype(np.int64)
    instance_keys = mint_detection_instance_keys(
        recording_identity=str(recording_identity),
        frame_indices=frame_indices,
        bbox_norm_coords=bbox_norm,
        class_ids=class_ids,
    )
    bbox_img, centers_img = derive_canonical_detection_geometry(
        bbox_norm,
        source_width=int(source_width),
        source_height=int(source_height),
    )
    counts = np.bincount(
        frame_indices.astype(np.int64, copy=False),
        minlength=selected_frames,
    )
    offsets = np.zeros(selected_frames + 1, dtype=np.int64)
    if selected_frames:
        offsets[1:] = np.cumsum(counts, dtype=np.int64)

    dimensions = CanonicalDetectionDimensions(
        n_frames=selected_frames,
        n_instances=stop,
        source_width=int(source_width),
        source_height=int(source_height),
    )
    arrays = {
        "instances/frame_indices": frame_indices,
        "instances/source_acquisition_frame_index": source_acquisition_frames,
        "instances/instance_key": instance_keys,
        "instances/bbox_norm_coords": bbox_norm,
        "instances/bbox_img_xyxy": bbox_img,
        "instances/centers_img_xy": centers_img,
        "instances/scores": scores,
        "instances/class_ids": class_ids,
        "instances/frame_row_offsets": offsets,
    }
    identity = {
        **dict(source_identity or {}),
        "recording_identity": str(recording_identity),
        "source_frame_count": total_frames,
        "selected_frame_count": selected_frames,
        "source_detection_rows": row_count,
        "selected_detection_rows": stop,
        "conversion": {
            "bbox_norm_coords": f"{source_bbox.dtype}->float32",
            "geometry_projection": "canonical_float32_exact",
            "source_acquisition_frame_index": "widened_frame_indices_identity",
            "frame_row_offsets": "cumsum_bincount_frame_indices",
            "instance_key": "minted_from_canonical_float32_bbox",
        },
    }
    return CanonicalDetectionBenchmarkInput(
        dimensions=dimensions,
        arrays=arrays,
        source_identity=identity,
    )


def load_detection_benchmark_input(
    source_group_path: Path,
    *,
    recording_identity: str,
    frame_limit: int | None,
) -> CanonicalDetectionBenchmarkInput:
    """Read one disposable legacy detection source group without mutation."""

    source_path = source_group_path.expanduser().resolve()
    metadata_path = source_path / "zarr.json"
    if not metadata_path.is_file():
        raise ValueError(f"Source is not a Zarr v3 group: {source_path}")
    source = zarr.open_group(
        str(source_path),
        mode="r",
        use_consolidated=False,
    )
    count_name = "frame_counts" if "frame_counts" in source else "n_detections"
    frame_count = int(source[count_name].shape[0])
    source_width = int(
        source.attrs.get("source_video_width") or source.attrs.get("source_full_width")
    )
    source_height = int(
        source.attrs.get("source_video_height")
        or source.attrs.get("source_full_height")
    )
    return build_canonical_detection_benchmark_input(
        source,
        recording_identity=recording_identity,
        frame_count=frame_count,
        source_width=source_width,
        source_height=source_height,
        frame_limit=frame_limit,
        source_identity={
            "source_group": str(source_path),
            "source_group_metadata_sha256": sha256_file(metadata_path),
            "source_open_mode": "read_only_direct_metadata",
        },
    )


__all__ = [
    "CanonicalDetectionBenchmarkInput",
    "build_canonical_detection_benchmark_input",
    "load_detection_benchmark_input",
]
