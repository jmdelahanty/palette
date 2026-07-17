"""Versioned source-video metadata and path resolution.

The v2 contract makes a recording-relative locator authoritative for ordinary
single-video recording Zarrs.  Existing absolute path attrs remain compatibility
mirrors during migration.  Legacy archives without the v2 schema keep a
well-defined fallback resolver, but conflicting populated mirrors fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional, Sequence


SOURCE_VIDEO_METADATA_SCHEMA_ID = "palette.source_video_metadata.v2"
SOURCE_VIDEO_LAYOUT_SINGLE = "single_video"
SOURCE_VIDEO_LOCATOR_RECORDING_RELATIVE = "recording_relative"
SOURCE_VIDEO_LOCATOR_ABSOLUTE = "absolute"


class SourceVideoMetadataError(ValueError):
    """Base error for invalid or unresolved source-video metadata."""


class SourceVideoMetadataConflictError(SourceVideoMetadataError):
    """Raised when canonical and compatibility video locators disagree."""


class SourceVideoMetadataMissingError(SourceVideoMetadataError):
    """Raised when an archive has no source-video locator."""


@dataclass(frozen=True)
class ResolvedSourceVideo:
    path: Path
    schema_id: Optional[str]
    layout: str
    locator_kind: str
    source: str
    compatibility_sources: tuple[str, ...]


def _norm_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _attrs(group: Any) -> Mapping[str, Any]:
    value = getattr(group, "attrs", None)
    return value if isinstance(value, Mapping) else {}


def _child(group: Any, name: str) -> Any:
    if group is None:
        return None
    try:
        if hasattr(group, "get"):
            return group.get(name)
        if name in group:
            return group[name]
    except Exception:
        return None
    return None


def _metadata_mapping(root_attrs: Mapping[str, Any]) -> Mapping[str, Any]:
    value = root_attrs.get("source_video_metadata")
    return value if isinstance(value, Mapping) else {}


def _recording_path(
    root_attrs: Mapping[str, Any],
    *,
    zarr_path: Optional[Path],
) -> Optional[Path]:
    declared = _norm_text(root_attrs.get("recording_path"))
    if declared:
        return Path(declared).expanduser().resolve()
    if zarr_path is None:
        return None
    resolved_zarr = Path(zarr_path).expanduser().resolve()
    if resolved_zarr.parent.name == "zarr":
        return resolved_zarr.parent.parent
    return None


def _resolve_recording_relative(recording_path: Path, relative_text: str) -> Path:
    relative = PurePosixPath(relative_text)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise SourceVideoMetadataError(
            f"Invalid recording-relative source-video path: {relative_text!r}"
        )
    base = recording_path.expanduser().resolve()
    candidate = (base / Path(*relative.parts)).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise SourceVideoMetadataError(
            f"Source-video locator escapes recording_path: {relative_text!r}"
        ) from exc
    return candidate


def _normalize_compatibility_path(
    raw: str,
    *,
    recording_path: Optional[Path],
    zarr_path: Optional[Path],
) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    if recording_path is not None:
        return (recording_path / path).resolve()
    if zarr_path is not None:
        return (Path(zarr_path).expanduser().resolve().parent / path).resolve()
    return path.resolve()


def _compatibility_candidates(
    root_attrs: Mapping[str, Any],
    raw_attrs: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any],
) -> list[tuple[str, str]]:
    candidates = (
        ("root.source_video_path", root_attrs.get("source_video_path")),
        ("root.source_path", root_attrs.get("source_path")),
        ("root.video_source_path", root_attrs.get("video_source_path")),
        ("source_video_metadata.source_path", metadata.get("source_path")),
        ("raw_video.source_path", raw_attrs.get("source_path")),
        ("raw_video.source_video_path", raw_attrs.get("source_video_path")),
    )
    return [
        (source, text)
        for source, value in candidates
        if (text := _norm_text(value)) is not None
    ]


def _validate_compatibility_candidates(
    candidates: Sequence[tuple[str, str]],
    *,
    expected: Path,
    recording_path: Optional[Path],
    zarr_path: Optional[Path],
) -> tuple[str, ...]:
    expected_resolved = expected.resolve()
    sources: list[str] = []
    conflicts: list[str] = []
    for source, raw in candidates:
        resolved = _normalize_compatibility_path(
            raw,
            recording_path=recording_path,
            zarr_path=zarr_path,
        )
        sources.append(source)
        if resolved != expected_resolved:
            conflicts.append(f"{source}={resolved}")
    if conflicts:
        raise SourceVideoMetadataConflictError(
            "Source-video compatibility paths disagree with the canonical locator "
            f"{expected_resolved}: {'; '.join(conflicts)}"
        )
    return tuple(sources)


def resolve_source_video_from_attrs(
    root_attrs: Mapping[str, Any],
    *,
    raw_video_attrs: Optional[Mapping[str, Any]] = None,
    zarr_path: Optional[Path] = None,
    require_exists: bool = False,
) -> ResolvedSourceVideo:
    """Resolve one source video from root/raw-video attribute mappings.

    Versioned metadata fails closed on malformed locators and conflicts. Legacy
    archives use the historical root/raw-video candidates but also fail when two
    populated mirrors identify different files. Multi-video/collection layouts
    intentionally require a collection-aware resolver.
    """

    metadata = _metadata_mapping(root_attrs)
    schema_id = _norm_text(metadata.get("schema_id"))
    recording_path = _recording_path(root_attrs, zarr_path=zarr_path)
    candidates = _compatibility_candidates(
        root_attrs,
        raw_video_attrs or {},
        metadata=metadata,
    )

    if schema_id == SOURCE_VIDEO_METADATA_SCHEMA_ID:
        layout = _norm_text(metadata.get("layout"))
        if layout != SOURCE_VIDEO_LAYOUT_SINGLE:
            raise SourceVideoMetadataError(
                f"{SOURCE_VIDEO_METADATA_SCHEMA_ID} requires layout="
                f"{SOURCE_VIDEO_LAYOUT_SINGLE!r}; got {layout!r}. Collection layouts "
                "must use their collection/frame-index resolver."
            )
        locator = metadata.get("locator")
        if not isinstance(locator, Mapping):
            raise SourceVideoMetadataError("source_video_metadata.locator must be an object")
        locator_kind = _norm_text(locator.get("kind"))
        if locator_kind == SOURCE_VIDEO_LOCATOR_RECORDING_RELATIVE:
            relative_path = _norm_text(locator.get("relative_path"))
            if recording_path is None:
                raise SourceVideoMetadataError(
                    "recording-relative source video requires root recording_path "
                    "or a Zarr located under <recording>/zarr/"
                )
            if relative_path is None:
                raise SourceVideoMetadataError(
                    "recording-relative source-video locator is missing relative_path"
                )
            resolved = _resolve_recording_relative(recording_path, relative_path)
        elif locator_kind == SOURCE_VIDEO_LOCATOR_ABSOLUTE:
            absolute_path = _norm_text(locator.get("path"))
            if absolute_path is None or not Path(absolute_path).expanduser().is_absolute():
                raise SourceVideoMetadataError(
                    "absolute source-video locator requires an absolute path"
                )
            resolved = Path(absolute_path).expanduser().resolve()
        else:
            raise SourceVideoMetadataError(
                f"Unsupported source-video locator kind: {locator_kind!r}"
            )
        compatibility_sources = _validate_compatibility_candidates(
            candidates,
            expected=resolved,
            recording_path=recording_path,
            zarr_path=zarr_path,
        )
        source = "source_video_metadata.locator"
        layout_value = layout
    else:
        if schema_id is not None:
            raise SourceVideoMetadataError(
                f"Unsupported source-video metadata schema: {schema_id!r}"
            )
        if not candidates:
            raise SourceVideoMetadataMissingError("No source-video locator is present")
        source, first_raw = candidates[0]
        resolved = _normalize_compatibility_path(
            first_raw,
            recording_path=recording_path,
            zarr_path=zarr_path,
        )
        compatibility_sources = _validate_compatibility_candidates(
            candidates,
            expected=resolved,
            recording_path=recording_path,
            zarr_path=zarr_path,
        )
        locator_kind = "legacy"
        layout_value = SOURCE_VIDEO_LAYOUT_SINGLE

    if require_exists and not resolved.is_file():
        raise SourceVideoMetadataError(f"Resolved source video does not exist: {resolved}")
    return ResolvedSourceVideo(
        path=resolved,
        schema_id=schema_id,
        layout=layout_value,
        locator_kind=str(locator_kind),
        source=source,
        compatibility_sources=compatibility_sources,
    )


def resolve_source_video(
    root: Any,
    *,
    zarr_path: Optional[Path] = None,
    require_exists: bool = False,
) -> ResolvedSourceVideo:
    """Resolve one source video from a Zarr-like root group."""

    return resolve_source_video_from_attrs(
        _attrs(root),
        raw_video_attrs=_attrs(_child(root, "raw_video")),
        zarr_path=zarr_path,
        require_exists=require_exists,
    )


def _infer_recording_path_from_source(source_path: Path) -> Optional[Path]:
    if source_path.parent.name == "cams":
        return source_path.parent.parent
    return None


def build_source_video_metadata_v2(
    metadata: Mapping[str, Any],
    *,
    recording_path: Optional[str | Path] = None,
    fingerprint_attrs: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Return v2 metadata while retaining ``source_path`` as a compatibility mirror."""

    payload = dict(metadata)
    source_text = _norm_text(payload.get("source_path"))
    if source_text is None:
        raise SourceVideoMetadataError("source video metadata is missing source_path")
    source_path = Path(source_text).expanduser().resolve()
    resolved_recording = (
        Path(recording_path).expanduser().resolve()
        if recording_path is not None
        else _infer_recording_path_from_source(source_path)
    )

    locator: dict[str, str]
    if resolved_recording is not None:
        try:
            relative = source_path.relative_to(resolved_recording)
        except ValueError:
            relative = None
        if relative is not None:
            locator = {
                "kind": SOURCE_VIDEO_LOCATOR_RECORDING_RELATIVE,
                "relative_path": relative.as_posix(),
            }
        else:
            locator = {
                "kind": SOURCE_VIDEO_LOCATOR_ABSOLUTE,
                "path": str(source_path),
            }
    else:
        locator = {
            "kind": SOURCE_VIDEO_LOCATOR_ABSOLUTE,
            "path": str(source_path),
        }

    payload.update(
        {
            "schema_id": SOURCE_VIDEO_METADATA_SCHEMA_ID,
            "layout": SOURCE_VIDEO_LAYOUT_SINGLE,
            "locator": locator,
            "source_path": str(source_path),
        }
    )
    fingerprint = dict(fingerprint_attrs or {})
    fingerprint_value = _norm_text(fingerprint.get("source_video_fingerprint"))
    if fingerprint_value:
        payload["file_fingerprint"] = {
            "strategy": _norm_text(
                fingerprint.get("source_video_fingerprint_strategy")
            ),
            "value": fingerprint_value,
            "size_bytes": fingerprint.get("source_video_size_bytes"),
            "mtime_ns": fingerprint.get("source_video_mtime_ns"),
            "relocation_stable": False,
        }
    return payload
