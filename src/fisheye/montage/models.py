"""Shared data models for montage selection, loading, and layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class MontageArtifactSpec:
    artifact_id: str
    label: str
    path: str
    visualization_contract_id: str | None = None


@dataclass(frozen=True)
class LoadedTile:
    artifact_id: str
    label: str
    path: str
    image: Image.Image | None
    error: str | None


@dataclass(frozen=True)
class RegistryRecording:
    recording_id: str
    zarr_path: Path
    dataset_id: str
    protocol_name: str | None
    arena_id: str | None
    recording_started_utc: str | None
    chaser_behaviors: tuple[str, ...] = ()
    chaser_count: int | None = None


@dataclass(frozen=True)
class MontageLayout:
    columns: int
    tile_width: int
    max_image_height: int
    margin: int = 24
    gutter: int = 18
    header_height: int = 92
    label_height: int = 30
