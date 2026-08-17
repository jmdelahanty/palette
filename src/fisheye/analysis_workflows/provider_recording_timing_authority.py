"""Exact read-only recording/timebase binding for Phase 4 provider offers.

The acquisition frame-clock publication remains the sole frame-domain
authority.  This module does not copy clock arrays and does not change the
nominal-FPS numerical semantics of existing provider-motion runs.  It binds
the selected immutable clock, the canonical source-video metadata, and the
recording identity into one canonical record that can be shared by position,
body-frame, motion, and temporal-selection consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from fisheye.shared.acquisition_frame_clock import (
    ACQUISITION_FRAME_CLOCK_SCHEMA_ID,
    AcquisitionFrameClockError,
    resolve_acquisition_frame_clock,
)
from fisheye.shared.json_safety import json_attr_safe_mapping
from fisheye.shared.source_video_metadata import (
    SOURCE_VIDEO_METADATA_SCHEMA_ID,
    SourceVideoMetadataError,
    resolve_source_video,
)
from fisheye.shared.zarr.manifest_digest import canonical_json_sha256
from fisheye.shared.zarr_io import open_zarr_root

PROVIDER_RECORDING_TIMING_AUTHORITY_SCHEMA_ID = (
    "palette.provider_recording_timing_authority"
)
PROVIDER_RECORDING_TIMING_AUTHORITY_SCHEMA_VERSION = 1
NOMINAL_FRAME_TIME_POLICY_ID = "nominal_fps_bound_to_acquisition_frame_domain.v1"


class ProviderRecordingTimingAuthorityError(ValueError):
    """Raised when a recording/timebase binding is absent, stale, or ambiguous."""


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _text(value: object, *, name: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ProviderRecordingTimingAuthorityError(
            f"{name} must be one exact nonempty string."
        )
    return value


def _positive_int(value: object, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ProviderRecordingTimingAuthorityError(
            f"{name} must be one positive exact integer."
        )
    return value


def _positive_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProviderRecordingTimingAuthorityError(
            f"{name} must be one positive finite number."
        )
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ProviderRecordingTimingAuthorityError(
            f"{name} must be one positive finite number."
        )
    return result


def _canonical_object(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProviderRecordingTimingAuthorityError(f"{name} must be one object.")
    try:
        canonical = dict(json_attr_safe_mapping(value))
        canonical_json_sha256(canonical)
    except (TypeError, ValueError) as exc:
        raise ProviderRecordingTimingAuthorityError(
            f"{name} must be strict canonical JSON."
        ) from exc
    return canonical


def _require_mirror(
    attrs: Mapping[str, Any],
    name: str,
    expected: int | float | str,
    *,
    owner: str,
) -> None:
    if name not in attrs:
        raise ProviderRecordingTimingAuthorityError(
            f"{owner}.{name} is required by the recording timing authority."
        )
    observed = attrs[name]
    if isinstance(expected, float):
        try:
            matches = not isinstance(observed, bool) and float(observed) == expected
        except (TypeError, ValueError):
            matches = False
    elif isinstance(expected, int):
        matches = type(observed) is int and observed == expected
    else:
        matches = type(observed) is str and observed == expected
    if not matches:
        raise ProviderRecordingTimingAuthorityError(
            f"{owner}.{name} differs from the canonical recording authority."
        )


@dataclass(frozen=True, init=False)
class ProviderRecordingTimingAuthority:
    """Loader-minted canonical recording and nominal-timebase authority."""

    analysis_zarr_path: Path
    record: Mapping[str, Any] = field(repr=False)
    sha256: str
    recording_id: str
    camera_id: str
    nominal_fps: float
    frame_count: int
    clock_run_path: str
    clock_record_sha256: str
    source_video_metadata_sha256: str
    _use_consolidated: bool = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(self, *, _verification_seal: object | None = None, **values: Any):
        if _verification_seal is not _AUTHORITY_SEAL:
            raise ProviderRecordingTimingAuthorityError(
                "Recording timing authorities must be minted by the strict loader."
            )
        for name, value in values.items():
            if name == "record":
                value = _freeze(value)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_seal", _AUTHORITY_SEAL)

    def validate_source_frame_indices(self, values: Any, *, name: str) -> np.ndarray:
        """Return an immutable validated snapshot in the complete frame domain."""

        self.assert_verified()
        array = np.asarray(values)
        if array.dtype != np.dtype("int64") or array.ndim != 1:
            raise ProviderRecordingTimingAuthorityError(
                f"{name} must be one exact int64 one-dimensional array."
            )
        if array.size and (
            int(array.min()) < 0 or int(array.max()) >= self.frame_count
        ):
            raise ProviderRecordingTimingAuthorityError(
                f"{name} contains an index outside the acquisition frame-clock domain."
            )
        result = np.array(array, dtype=np.int64, copy=True, order="C")
        result.setflags(write=False)
        return result

    def assert_verified(self) -> None:
        if self._seal is not _AUTHORITY_SEAL:
            raise ProviderRecordingTimingAuthorityError(
                "Recording timing authority verification seal is absent."
            )
        if canonical_json_sha256(_thaw(self.record)) != self.sha256:
            raise ProviderRecordingTimingAuthorityError(
                "Recording timing authority digest is stale."
            )

    def assert_current(self) -> None:
        current = load_provider_recording_timing_authority(
            self.analysis_zarr_path,
            required=True,
            use_consolidated=self._use_consolidated,
            expected_sha256=self.sha256,
        )
        assert current is not None


_AUTHORITY_SEAL = object()


def _load_once(
    archive: Path,
    *,
    use_consolidated: bool,
) -> ProviderRecordingTimingAuthority | None:
    try:
        root = open_zarr_root(
            archive,
            mode="r",
            use_consolidated=use_consolidated,
        )
    except (KeyError, TypeError, ValueError, OSError, RuntimeError) as exc:
        metadata_mode = "consolidated" if use_consolidated else "direct"
        raise ProviderRecordingTimingAuthorityError(
            f"Unable to open {metadata_mode} recording timing metadata: {exc}"
        ) from exc
    try:
        clock = resolve_acquisition_frame_clock(root, required=False)
    except (AcquisitionFrameClockError, KeyError, TypeError, ValueError) as exc:
        raise ProviderRecordingTimingAuthorityError(str(exc)) from exc
    if clock is None:
        return None

    attrs = getattr(root, "attrs", {})
    recording_id = _text(attrs.get("recording_id"), name="root.recording_id")
    metadata = _canonical_object(
        attrs.get("source_video_metadata"), name="root.source_video_metadata"
    )
    if metadata.get("schema_id") != SOURCE_VIDEO_METADATA_SCHEMA_ID:
        raise ProviderRecordingTimingAuthorityError(
            "The recording requires canonical source_video_metadata.v2."
        )
    if metadata.get("layout") != "single_video":
        raise ProviderRecordingTimingAuthorityError(
            "The recording timing authority currently supports single-video layout only."
        )
    try:
        resolved_video = resolve_source_video(
            root,
            zarr_path=archive,
            require_exists=False,
        )
    except SourceVideoMetadataError as exc:
        raise ProviderRecordingTimingAuthorityError(str(exc)) from exc
    if resolved_video.schema_id != SOURCE_VIDEO_METADATA_SCHEMA_ID:
        raise ProviderRecordingTimingAuthorityError(
            "The resolved source video does not use canonical source_video_metadata.v2."
        )
    camera_id = _text(metadata.get("camera_id"), name="source_video_metadata.camera_id")
    if camera_id != clock.camera_id:
        raise ProviderRecordingTimingAuthorityError(
            "Source-video and acquisition-clock camera identities differ."
        )
    nominal_fps = _positive_float(metadata.get("fps"), name="source_video_metadata.fps")
    frame_count = _positive_int(
        metadata.get("total_frames"), name="source_video_metadata.total_frames"
    )
    if frame_count != clock.row_count:
        raise ProviderRecordingTimingAuthorityError(
            "Source-video frame count differs from the acquisition clock."
        )
    _require_mirror(attrs, "recording_id", recording_id, owner="root")
    _require_mirror(attrs, "camera_id", camera_id, owner="root")
    _require_mirror(attrs, "fps", nominal_fps, owner="root")
    _require_mirror(attrs, "total_frames", frame_count, owner="root")

    raw_video = root.get("raw_video")
    if raw_video is not None:
        raw_attrs = getattr(raw_video, "attrs", {})
        _require_mirror(raw_attrs, "fps", nominal_fps, owner="raw_video")
        _require_mirror(raw_attrs, "total_frames", frame_count, owner="raw_video")

    clock_run = root[clock.group_path]
    parent_frame_index = np.asarray(clock_run["parent_frame_index"][:])
    if parent_frame_index.dtype != np.dtype("int64") or not np.array_equal(
        parent_frame_index, np.arange(frame_count, dtype=np.int64)
    ):
        raise ProviderRecordingTimingAuthorityError(
            "Acquisition clock does not expose the complete zero-based frame domain."
        )

    source_metadata_sha256 = canonical_json_sha256(metadata)
    array_sha256 = _canonical_object(
        clock.record.get("array_sha256"), name="acquisition clock array digests"
    )
    record = {
        "schema_id": PROVIDER_RECORDING_TIMING_AUTHORITY_SCHEMA_ID,
        "schema_version": PROVIDER_RECORDING_TIMING_AUTHORITY_SCHEMA_VERSION,
        "policy_id": NOMINAL_FRAME_TIME_POLICY_ID,
        "recording_id": recording_id,
        "camera_id": camera_id,
        "nominal_fps": nominal_fps,
        "frame_count": frame_count,
        "acquisition_frame_clock": {
            "schema_id": ACQUISITION_FRAME_CLOCK_SCHEMA_ID,
            "run_path": clock.group_path,
            "record_sha256": clock.record_sha256,
            "array_sha256": array_sha256,
        },
        "source_video_metadata": {
            "schema_id": SOURCE_VIDEO_METADATA_SCHEMA_ID,
            "sha256": source_metadata_sha256,
        },
        "numerical_semantics": {
            "timebase": "nominal_fps",
            "frame_delta_rule": "acquisition_frame_index_difference_divided_by_fps",
            "camera_timestamp_arrays_copied": False,
        },
    }
    digest = canonical_json_sha256(record)
    return ProviderRecordingTimingAuthority(
        analysis_zarr_path=archive,
        record=record,
        sha256=digest,
        recording_id=recording_id,
        camera_id=camera_id,
        nominal_fps=nominal_fps,
        frame_count=frame_count,
        clock_run_path=clock.group_path,
        clock_record_sha256=clock.record_sha256,
        source_video_metadata_sha256=source_metadata_sha256,
        _use_consolidated=use_consolidated,
        _verification_seal=_AUTHORITY_SEAL,
    )


def load_provider_recording_timing_authority(
    analysis_zarr: str | Path,
    *,
    required: bool = True,
    use_consolidated: bool = True,
    expected_sha256: str | None = None,
) -> ProviderRecordingTimingAuthority | None:
    """Load the current exact recording/timebase binding without writing data."""

    if type(required) is not bool or type(use_consolidated) is not bool:
        raise ProviderRecordingTimingAuthorityError(
            "required and use_consolidated must be exact booleans."
        )
    archive = Path(analysis_zarr).expanduser().resolve()
    authority = _load_once(archive, use_consolidated=use_consolidated)
    if authority is None:
        if required:
            raise ProviderRecordingTimingAuthorityError(
                "The analysis archive has no acquisition frame-clock authority."
            )
        return None
    if expected_sha256 is not None and authority.sha256 != expected_sha256:
        raise ProviderRecordingTimingAuthorityError(
            "The recording timing authority differs from the expected digest."
        )
    if use_consolidated:
        direct = _load_once(archive, use_consolidated=False)
        if direct is None or direct.sha256 != authority.sha256:
            raise ProviderRecordingTimingAuthorityError(
                "Published consolidated recording timing authority differs from direct metadata."
            )
    return authority


__all__ = [
    "NOMINAL_FRAME_TIME_POLICY_ID",
    "PROVIDER_RECORDING_TIMING_AUTHORITY_SCHEMA_ID",
    "PROVIDER_RECORDING_TIMING_AUTHORITY_SCHEMA_VERSION",
    "ProviderRecordingTimingAuthority",
    "ProviderRecordingTimingAuthorityError",
    "load_provider_recording_timing_authority",
]
