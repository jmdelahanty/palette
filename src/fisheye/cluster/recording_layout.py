"""Layout-neutral recording targets for production workflow planning.

These models describe scientific recording identity separately from the
physical partitioning of its source video.  They do not inspect array payloads,
submit jobs, or decide how work is packaged by LSF.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


class RecordingLayout(str, Enum):
    """Supported physical source layouts for one scientific recording."""

    WHOLE_VIDEO = "whole_video"
    CLIPPED_COLLECTION = "clipped_collection"


class FrameMappingMode(str, Enum):
    """How work-unit-local frames map to canonical recording frames."""

    IDENTITY = "identity"
    RECORDING_FRAME_INDEX = "recording_frame_index"


@dataclass(frozen=True)
class VideoFrameMapping:
    """Authority for mapping one work unit onto the recording timeline."""

    mode: FrameMappingMode
    recording_frame_index: Path | None = None
    canonical_start_frame: int | None = None

    def __post_init__(self) -> None:
        mode = self.mode
        if not isinstance(mode, FrameMappingMode):
            mode = FrameMappingMode(mode)
            object.__setattr__(self, "mode", mode)
        if self.recording_frame_index is not None:
            object.__setattr__(
                self,
                "recording_frame_index",
                self.recording_frame_index.expanduser().resolve(),
            )
        if self.canonical_start_frame is not None:
            object.__setattr__(
                self,
                "canonical_start_frame",
                int(self.canonical_start_frame),
            )

        if mode is FrameMappingMode.IDENTITY:
            if self.recording_frame_index is not None:
                raise ValueError(
                    "Identity frame mapping cannot declare a recording-frame index."
                )
            if self.canonical_start_frame != 0:
                raise ValueError(
                    "Whole-video identity mapping must start at canonical frame 0."
                )
        elif mode is FrameMappingMode.RECORDING_FRAME_INDEX:
            if self.recording_frame_index is None:
                raise ValueError(
                    "Indexed frame mapping requires recording_frame_index."
                )
            if self.canonical_start_frame is not None:
                raise ValueError(
                    "Indexed frame mapping cannot also declare a positional offset."
                )

    @classmethod
    def identity(cls) -> VideoFrameMapping:
        return cls(
            mode=FrameMappingMode.IDENTITY,
            canonical_start_frame=0,
        )

    @classmethod
    def indexed(cls, recording_frame_index: Path) -> VideoFrameMapping:
        return cls(
            mode=FrameMappingMode.RECORDING_FRAME_INDEX,
            recording_frame_index=recording_frame_index,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "recording_frame_index": (
                str(self.recording_frame_index)
                if self.recording_frame_index is not None
                else None
            ),
            "canonical_start_frame": self.canonical_start_frame,
        }


@dataclass(frozen=True)
class VideoWorkUnit:
    """One independently addressable physical video partition."""

    work_unit_id: str
    source_partition_id: str
    source_partition_index: int
    video_path: Path
    camera_serial: str
    frame_mapping: VideoFrameMapping
    frame_count: int | None = None
    arena_id: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("work_unit_id", "source_partition_id", "camera_serial"):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"Video work-unit {field_name} cannot be empty.")
            object.__setattr__(self, field_name, value)
        object.__setattr__(
            self,
            "source_partition_index",
            int(self.source_partition_index),
        )
        if self.source_partition_index < 0:
            raise ValueError("Video work-unit partition index cannot be negative.")
        if str(self.video_path).strip() in {"", "."}:
            raise ValueError("Video work-unit video_path cannot be empty.")
        object.__setattr__(self, "video_path", self.video_path.expanduser().resolve())
        if not isinstance(self.frame_mapping, VideoFrameMapping):
            raise TypeError("Video work-unit frame_mapping must be VideoFrameMapping.")
        if self.frame_count is not None:
            object.__setattr__(self, "frame_count", int(self.frame_count))
            if self.frame_count <= 0:
                raise ValueError("Video work-unit frame_count must be positive.")
        if self.arena_id is not None:
            arena = str(self.arena_id).strip()
            object.__setattr__(self, "arena_id", arena or None)

    def to_json(self) -> dict[str, Any]:
        return {
            "work_unit_id": self.work_unit_id,
            "source_partition_id": self.source_partition_id,
            "source_partition_index": self.source_partition_index,
            "video_path": str(self.video_path),
            "camera_serial": self.camera_serial,
            "frame_count": self.frame_count,
            "arena_id": self.arena_id,
            "frame_mapping": self.frame_mapping.to_json(),
        }


@dataclass(frozen=True)
class RecordingTarget:
    """One scientific recording and its complete physical video work units."""

    target_id: str
    recording_id: str
    recording_dir: Path
    analysis_zarr: Path
    layout: RecordingLayout
    work_units: tuple[VideoWorkUnit, ...]
    expected_subject_count: int = 1

    def __post_init__(self) -> None:
        for field_name in ("target_id", "recording_id"):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"Recording target {field_name} cannot be empty.")
            object.__setattr__(self, field_name, value)
        object.__setattr__(
            self,
            "recording_dir",
            self.recording_dir.expanduser().resolve(),
        )
        object.__setattr__(
            self,
            "analysis_zarr",
            self.analysis_zarr.expanduser().resolve(),
        )
        layout = self.layout
        if not isinstance(layout, RecordingLayout):
            layout = RecordingLayout(layout)
            object.__setattr__(self, "layout", layout)
        work_units = tuple(self.work_units)
        object.__setattr__(self, "work_units", work_units)
        object.__setattr__(
            self,
            "expected_subject_count",
            int(self.expected_subject_count),
        )
        if self.expected_subject_count <= 0:
            raise ValueError("Recording target expected_subject_count must be positive.")
        if not work_units:
            raise ValueError("Recording target requires at least one video work unit.")
        if any(not isinstance(unit, VideoWorkUnit) for unit in work_units):
            raise TypeError("Recording target work_units must be VideoWorkUnit values.")

        work_unit_ids = [unit.work_unit_id for unit in work_units]
        if len(set(work_unit_ids)) != len(work_unit_ids):
            raise ValueError("Recording target work_unit_id values must be unique.")
        partition_keys = [
            (
                unit.source_partition_id,
                unit.source_partition_index,
                unit.camera_serial,
            )
            for unit in work_units
        ]
        if len(set(partition_keys)) != len(partition_keys):
            raise ValueError("Recording target source partitions must be unique.")

        if layout is RecordingLayout.WHOLE_VIDEO:
            if len(work_units) != 1:
                raise ValueError("Whole-video targets require exactly one work unit.")
            if work_units[0].frame_mapping.mode is not FrameMappingMode.IDENTITY:
                raise ValueError("Whole-video targets require identity frame mapping.")
        elif layout is RecordingLayout.CLIPPED_COLLECTION:
            if any(
                unit.frame_mapping.mode is not FrameMappingMode.RECORDING_FRAME_INDEX
                for unit in work_units
            ):
                raise ValueError(
                    "Clipped-collection targets require indexed frame mappings."
                )
            authorities = {
                unit.frame_mapping.recording_frame_index for unit in work_units
            }
            if len(authorities) != 1:
                raise ValueError(
                    "Clipped-collection work units must share one recording-frame index."
                )

    @property
    def recording_frame_index(self) -> Path | None:
        if self.layout is RecordingLayout.WHOLE_VIDEO:
            return None
        return self.work_units[0].frame_mapping.recording_frame_index

    def to_json(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "recording_id": self.recording_id,
            "recording_dir": str(self.recording_dir),
            "analysis_zarr": str(self.analysis_zarr),
            "layout": self.layout.value,
            "expected_subject_count": self.expected_subject_count,
            "work_unit_count": len(self.work_units),
            "work_units": [unit.to_json() for unit in self.work_units],
        }


def clipped_recording_target(
    *,
    target_id: str,
    recording_id: str,
    recording_dir: Path,
    analysis_zarr: Path,
    work_units: Sequence[Mapping[str, Any]],
    recording_frame_index: Path | None = None,
    expected_subject_count: int = 1,
) -> RecordingTarget:
    """Adapt existing clipped-plan work units to the neutral target contract."""

    resolved_recording_dir = recording_dir.expanduser().resolve()
    frame_index = (
        recording_frame_index.expanduser().resolve()
        if recording_frame_index is not None
        else resolved_recording_dir / "recording_frame_index.parquet"
    )
    units: list[VideoWorkUnit] = []
    for raw in work_units:
        source = raw.get("source")
        if not isinstance(source, Mapping):
            raise ValueError("Clipped work unit requires a source mapping.")
        clip_id = str(raw.get("clip_id") or "").strip()
        if not clip_id:
            raise ValueError("Clipped work unit requires clip_id.")
        video_path = str(source.get("video_path") or "").strip()
        if not video_path:
            raise ValueError("Clipped work unit requires source.video_path.")
        frame_count_raw = raw.get("frame_count")
        units.append(
            VideoWorkUnit(
                work_unit_id=str(raw.get("work_unit_id") or ""),
                source_partition_id=clip_id,
                source_partition_index=int(raw.get("clip_index")),
                video_path=Path(video_path),
                camera_serial=str(raw.get("camera_serial") or ""),
                frame_count=(
                    int(frame_count_raw) if frame_count_raw is not None else None
                ),
                arena_id=(
                    str(raw.get("arena_id"))
                    if raw.get("arena_id") is not None
                    else None
                ),
                frame_mapping=VideoFrameMapping.indexed(frame_index),
            )
        )
    return RecordingTarget(
        target_id=target_id,
        recording_id=recording_id,
        recording_dir=resolved_recording_dir,
        analysis_zarr=analysis_zarr,
        layout=RecordingLayout.CLIPPED_COLLECTION,
        work_units=tuple(units),
        expected_subject_count=expected_subject_count,
    )


def whole_video_recording_target(
    *,
    target_id: str,
    recording_id: str,
    recording_dir: Path,
    analysis_zarr: Path,
    video_path: Path,
    camera_serial: str,
    frame_count: int | None = None,
    arena_id: str | None = None,
    expected_subject_count: int = 1,
    work_unit_id: str | None = None,
) -> RecordingTarget:
    """Build the one-work-unit adapter for a canonical whole video."""

    unit = VideoWorkUnit(
        work_unit_id=work_unit_id or f"{target_id}:whole_video",
        source_partition_id="whole_video",
        source_partition_index=0,
        video_path=video_path,
        camera_serial=camera_serial,
        frame_count=frame_count,
        arena_id=arena_id,
        frame_mapping=VideoFrameMapping.identity(),
    )
    return RecordingTarget(
        target_id=target_id,
        recording_id=recording_id,
        recording_dir=recording_dir,
        analysis_zarr=analysis_zarr,
        layout=RecordingLayout.WHOLE_VIDEO,
        work_units=(unit,),
        expected_subject_count=expected_subject_count,
    )


__all__ = [
    "FrameMappingMode",
    "RecordingLayout",
    "RecordingTarget",
    "VideoFrameMapping",
    "VideoWorkUnit",
    "clipped_recording_target",
    "whole_video_recording_target",
]
