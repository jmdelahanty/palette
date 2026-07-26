#!/usr/bin/env python3
"""Migrate metadata-only archives to sealed external-video acquisition authority.

The command is dry-run by default.  It is deliberately limited to one-video
recordings whose camera identity can be corroborated by recording attrs, the
recording id, and the source-video filename.  Video geometry is probed from the
source file and must agree with every populated legacy metadata field before an
archive is changed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import zarr

from fisheye.shared.import_video_metadata import (
    probe_video_metadata,
    publish_external_video_acquisition_authority,
    write_video_metadata,
)
from fisheye.shared.pixel_frame_authority import (
    load_persisted_acquisition_camera_authority,
)
from fisheye.shared.source_video_metadata import resolve_source_video
from fisheye.shared.zarr_helpers import consolidate_metadata_capture_expected_warnings


MIGRATION_SCHEMA_ID = "palette.external_video_acquisition_migration.v1"
_CAMERA_TOKEN = re.compile(r"(?i)(?:^|[^a-z0-9])cam(?P<camera_id>[0-9]+)(?:[^0-9]|$)")


@dataclass(frozen=True)
class MigrationPlan:
    zarr_path: str
    recording_id: str
    camera_id: str
    source_video_path: str
    width: int
    height: int
    total_frames: int
    fps: float
    status: str
    authority_path: str | None = None


def _open_group(path: Path, *, mode: str) -> zarr.Group:
    try:
        return zarr.open_group(str(path), mode=mode, use_consolidated=False)
    except TypeError:
        return zarr.open_group(str(path), mode=mode, consolidated=False)


def _required_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field} must be a nonempty canonical string.")
    return value


def _camera_token(value: str, *, field: str) -> str:
    match = _CAMERA_TOKEN.search(value)
    if match is None:
        raise ValueError(f"{field} does not contain a canonical Cam<serial> token: {value!r}")
    return match.group("camera_id")


def _recording_path(root: zarr.Group, zarr_path: Path) -> Path:
    inferred = zarr_path.resolve().parent.parent
    declared_raw = root.attrs.get("recording_path")
    if declared_raw not in (None, ""):
        declared = Path(str(declared_raw)).expanduser().resolve()
        if declared != inferred:
            raise ValueError(
                f"recording_path conflicts with the Zarr location: {declared} != {inferred}"
            )
    if inferred.name == "zarr":
        raise ValueError("Expected archive layout <recording>/zarr/<archive>.zarr.")
    return inferred


def _camera_id(root: zarr.Group, *, recording_id: str, source_video: Path) -> str:
    serials_raw = root.attrs.get("camera_serials")
    if not isinstance(serials_raw, (list, tuple)) or len(serials_raw) != 1:
        raise ValueError("camera_serials must contain exactly one camera for this migration.")
    serial = _required_text(str(serials_raw[0]), field="camera_serials[0]")
    candidates = {
        "camera_serials[0]": serial,
        "recording_id": _camera_token(recording_id, field="recording_id"),
        "source_video filename": _camera_token(source_video.name, field="source_video filename"),
    }
    existing = root.attrs.get("camera_id")
    if existing not in (None, ""):
        candidates["camera_id"] = _required_text(existing, field="camera_id")
    if len(set(candidates.values())) != 1:
        raise ValueError(f"Camera identity evidence conflicts: {candidates}")
    return serial


def _source_video(root: zarr.Group, *, zarr_path: Path, recording_path: Path) -> Path:
    try:
        candidate = resolve_source_video(root, zarr_path=zarr_path, require_exists=True).path
    except Exception:
        candidates = sorted((recording_path / "cams").glob("*.mp4"))
        if len(candidates) != 1:
            raise ValueError(
                "Archive has no resolvable source-video locator and recording/cams "
                f"contains {len(candidates)} MP4 files; exactly one is required."
            )
        candidate = candidates[0].resolve()
    expected_parent = (recording_path / "cams").resolve()
    if candidate.resolve().parent != expected_parent:
        raise ValueError(f"Source video is not the recording's direct cams/ child: {candidate}")
    return candidate.resolve()


def _positive_int(value: Any, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"Probed {field} must be an exact positive integer; got {value!r}.")
    return value


def _positive_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Probed {field} must be positive; got {value!r}.")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"Probed {field} must be positive; got {value!r}.")
    return result


def _assert_legacy_agreement(root: zarr.Group, meta: Mapping[str, Any]) -> None:
    source_metadata = root.attrs.get("source_video_metadata")
    nested = source_metadata if isinstance(source_metadata, Mapping) else {}
    for field in ("width", "height", "total_frames"):
        expected = meta[field]
        for label, attrs in (("root", root.attrs), ("source_video_metadata", nested)):
            value = attrs.get(field)
            if value is not None and int(value) != expected:
                raise ValueError(f"{label}.{field} conflicts with probed video: {value} != {expected}")
    for label, attrs in (("root", root.attrs), ("source_video_metadata", nested)):
        value = attrs.get("fps")
        if value is not None and not math.isclose(float(value), meta["fps"], rel_tol=0, abs_tol=1e-6):
            raise ValueError(f"{label}.fps conflicts with probed video: {value} != {meta['fps']}")


def plan_migration(zarr_path: Path) -> tuple[MigrationPlan, dict[str, Any]]:
    zarr_path = Path(zarr_path).expanduser().resolve()
    root = _open_group(zarr_path, mode="r")
    recording_id = _required_text(root.attrs.get("recording_id"), field="recording_id")
    recording_path = _recording_path(root, zarr_path)
    source_video = _source_video(
        root,
        zarr_path=zarr_path,
        recording_path=recording_path,
    )
    camera_id = _camera_id(root, recording_id=recording_id, source_video=source_video)
    meta = dict(probe_video_metadata(source_video))
    meta.update(
        {
            "width": _positive_int(meta.get("width"), field="width"),
            "height": _positive_int(meta.get("height"), field="height"),
            "total_frames": _positive_int(meta.get("total_frames"), field="total_frames"),
            "fps": _positive_float(meta.get("fps"), field="fps"),
            "camera_id": camera_id,
        }
    )
    _assert_legacy_agreement(root, meta)

    status = "would_migrate_and_seal"
    authority_path: str | None = None
    try:
        _ownership, authority = load_persisted_acquisition_camera_authority(
            root,
            expected_camera_id=camera_id,
        )
    except Exception:
        pass
    else:
        if authority.record.width_px != meta["width"] or authority.record.height_px != meta["height"]:
            raise ValueError("Existing acquisition authority conflicts with probed video geometry.")
        status = "already_sealed"
        authority_path = authority.record_ref.split("@", 1)[0].lstrip("/")

    return (
        MigrationPlan(
            zarr_path=str(zarr_path),
            recording_id=recording_id,
            camera_id=camera_id,
            source_video_path=str(source_video),
            width=meta["width"],
            height=meta["height"],
            total_frames=meta["total_frames"],
            fps=meta["fps"],
            status=status,
            authority_path=authority_path,
        ),
        meta,
    )


def apply_migration(
    zarr_path: Path,
    *,
    consolidate_metadata: bool = True,
) -> MigrationPlan:
    plan, meta = plan_migration(zarr_path)
    if plan.status == "already_sealed":
        return plan
    root = _open_group(Path(plan.zarr_path), mode="a")
    root.attrs["camera_id"] = plan.camera_id
    write_video_metadata(
        root,
        meta,
        overwrite=True,
        import_purpose="analysis",
        recording_path=Path(plan.zarr_path).parent.parent,
    )
    publication = publish_external_video_acquisition_authority(root)
    _ownership, verified = load_persisted_acquisition_camera_authority(
        root,
        expected_camera_id=plan.camera_id,
    )
    if verified.record.width_px != plan.width or verified.record.height_px != plan.height:
        raise RuntimeError("Published acquisition authority failed geometry readback.")
    if consolidate_metadata:
        consolidate_metadata_capture_expected_warnings(Path(plan.zarr_path))
    return MigrationPlan(
        **{
            **asdict(plan),
            "status": "migrated_and_sealed",
            "authority_path": publication["authority_path"],
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_paths", nargs="+", type=Path)
    parser.add_argument("--apply", action="store_true", help="Write and seal after preflight.")
    parser.add_argument("--no-consolidate", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit one JSON object per archive.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    failed = False
    for zarr_path in args.zarr_paths:
        try:
            if args.apply:
                result = apply_migration(
                    zarr_path,
                    consolidate_metadata=not args.no_consolidate,
                )
            else:
                result, _ = plan_migration(zarr_path)
            payload = {"schema_id": MIGRATION_SCHEMA_ID, **asdict(result)}
        except Exception as exc:
            failed = True
            payload = {
                "schema_id": MIGRATION_SCHEMA_ID,
                "zarr_path": str(Path(zarr_path).expanduser().resolve()),
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        if args.json:
            print(json.dumps(payload, sort_keys=True))
        else:
            print(f"{payload['status']}\t{payload['zarr_path']}\t{payload.get('error', '')}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
