"""Resolve canonical acquisition authority for native detection workflows."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Protocol

import pyarrow.parquet as pq

from fisheye.cluster.native_detection import NativeDetectionAuthoritySpec
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256


class NativeDetectionTarget(Protocol):
    recording_id: str
    analysis_zarr: Path


@dataclass(frozen=True)
class NativeArchiveAuthority:
    recording_identity: str
    camera_serial: str
    n_frames: int
    source_width: int
    source_height: int
    frame: NativeDetectionAuthoritySpec
    pixel: NativeDetectionAuthoritySpec

    def to_json(self) -> dict[str, object]:
        return {
            "recording_identity": self.recording_identity,
            "camera_serial": self.camera_serial,
            "n_frames": self.n_frames,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "source_frame_authority": {
                "record_ref": self.frame.record_ref,
                "record_sha256": self.frame.record_sha256,
            },
            "source_pixel_authority": {
                "record_ref": self.pixel.record_ref,
                "record_sha256": self.pixel.record_sha256,
            },
            "pixel_authority_basis": "acquisition_camera_frame.width_px_height_px",
        }


def _zarr_attrs(path: Path) -> Mapping[str, Any]:
    # Historical roots may carry unrelated NaN/Infinity values. The selected
    # acquisition record itself is validated through canonical_json_sha256.
    with (path / "zarr.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a Zarr JSON object at {path}.")
    attrs = payload.get("attributes")
    if not isinstance(attrs, Mapping):
        raise ValueError(f"Zarr node has no attributes mapping: {path}")
    return attrs


def load_native_archive_authority(
    target: NativeDetectionTarget,
) -> NativeArchiveAuthority:
    """Resolve the immutable acquisition record used for time and pixels."""

    archive = target.analysis_zarr.expanduser().resolve()
    root_attrs = _zarr_attrs(archive)
    recording_identity = str(root_attrs.get("recording_id") or "").strip()
    if not recording_identity or recording_identity != target.recording_id:
        raise ValueError(
            "Target recording_id differs from the analysis archive identity: "
            f"target={target.recording_id!r}, archive={recording_identity!r}."
        )
    raw_attrs = _zarr_attrs(archive / "raw_video")
    status = raw_attrs.get("acquisition_authority_publication_status")
    if not isinstance(status, Mapping):
        raise ValueError("Analysis archive lacks acquisition publication status.")
    if status.get("status") != "published_canonical_v1":
        raise ValueError("Acquisition authority is not published canonical v1.")
    authority_path = str(status.get("authority_path") or "").strip().strip("/")
    expected_prefix = "analysis/acquisition_camera_frames/"
    if not authority_path.startswith(expected_prefix):
        raise ValueError("Acquisition authority path is not canonical.")
    camera_serial = authority_path.removeprefix(expected_prefix)
    if not camera_serial or "/" in camera_serial:
        raise ValueError("Acquisition authority camera serial is invalid.")
    authority_attrs = _zarr_attrs(archive / authority_path)
    record = authority_attrs.get("acquisition_camera_frame")
    if not isinstance(record, Mapping):
        raise ValueError("Canonical acquisition-camera record is missing.")
    digest = str(authority_attrs.get("acquisition_camera_frame_sha256") or "")
    if digest != canonical_json_sha256(record):
        raise ValueError("Acquisition-camera record digest is stale.")
    if record.get("recording_id") != recording_identity:
        raise ValueError("Acquisition-camera record has the wrong recording identity.")
    if str(record.get("camera_id") or "") != camera_serial:
        raise ValueError("Acquisition-camera record has the wrong camera serial.")
    n_frames = int(record.get("source_total_frames") or 0)
    width = int(record.get("width_px") or 0)
    height = int(record.get("height_px") or 0)
    if min(n_frames, width, height) <= 0:
        raise ValueError("Acquisition-camera record has invalid source dimensions.")
    if int(record.get("frame_count") or 0) != n_frames:
        raise ValueError("Acquisition-camera frame count disagrees with source extent.")
    pointer = NativeDetectionAuthoritySpec(
        record_ref=f"/{authority_path}@acquisition_camera_frame",
        record_sha256=digest,
    )
    return NativeArchiveAuthority(
        recording_identity=recording_identity,
        camera_serial=camera_serial,
        n_frames=n_frames,
        source_width=width,
        source_height=height,
        frame=pointer,
        pixel=pointer,
    )


def validate_recording_frame_index(path: Path, *, n_frames: int) -> None:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"recording_frame_index.parquet not found: {resolved}")
    parquet = pq.ParquetFile(resolved)
    required = {
        "camera_serial",
        "clip_id",
        "clip_local_frame_index",
        "parent_frame_index",
    }
    missing = sorted(required - set(parquet.schema_arrow.names))
    if missing:
        raise ValueError(f"recording frame index is missing columns: {missing}")
    if parquet.metadata.num_rows != int(n_frames):
        raise ValueError(
            "recording frame index row count differs from acquisition authority: "
            f"{parquet.metadata.num_rows} != {n_frames}."
        )


__all__ = [
    "NativeArchiveAuthority",
    "load_native_archive_authority",
    "validate_recording_frame_index",
]
