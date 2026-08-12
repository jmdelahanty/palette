#!/usr/bin/env python3
"""Audit or repair canonical external-video acquisition authority.

The command is dry-run by default.  It never decodes or copies source video.
Before publishing authority it requires the canonical source-video locator and
persisted ``stat_v1`` fingerprint to match the live file exactly.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import zarr

from fisheye.registry.db import RegistryPaths
from fisheye.shared.acquisition_publication_status import (
    ACQUISITION_AUTHORITY_PENDING,
    ACQUISITION_AUTHORITY_PUBLISHED,
    ACQUISITION_AUTHORITY_STATUS_ATTR,
    EXTERNAL_ACQUISITION_AUTHORITY_MODE,
    load_acquisition_authority_publication_status,
)
from fisheye.shared.import_source_fingerprint import source_stat_fingerprint_attrs
from fisheye.shared.import_video_metadata import (
    publish_external_video_acquisition_authority,
)
from fisheye.shared.json_safety import write_json_atomic
from fisheye.shared.pixel_frame_authority import (
    PixelFrameAuthorityError,
    load_persisted_acquisition_camera_authority,
    parse_source_video_metadata,
)
from fisheye.shared.source_video_metadata import resolve_source_video
from fisheye.shared.zarr_discovery import (
    discover_filesystem_zarrs,
    discover_registry_zarrs,
    load_path_list,
)
from fisheye.shared.zarr_helpers import open_zarr_group_direct
from fisheye.shared.zarr_helpers import (
    consolidate_metadata_capture_expected_warnings,
)

SCHEMA_ID = "palette.external_video_acquisition_authority_repair.v1"


def _remove_legacy_imageio_metadata(
    root: zarr.Group,
    *,
    canonical_source_video_metadata: Mapping[str, Any],
) -> None:
    """Remove redundant legacy probe metadata after authority validation."""

    root_attrs = copy.deepcopy(dict(root.attrs))
    root_attrs.pop("imageio_metadata", None)
    root_attrs["source_video_metadata"] = copy.deepcopy(
        dict(canonical_source_video_metadata)
    )
    root.attrs.put(root_attrs)
    reparsed = parse_source_video_metadata(root.attrs.get("source_video_metadata"))
    if (
        reparsed != dict(canonical_source_video_metadata)
        or "imageio_metadata" in root.attrs
        or "imageio_metadata" in root.attrs["source_video_metadata"]
    ):
        raise PixelFrameAuthorityError(
            "Legacy ImageIO metadata removal did not round-trip exactly."
        )


def _expected_live_fingerprint(
    video_path: Path, metadata: dict[str, Any]
) -> dict[str, Any]:
    live = source_stat_fingerprint_attrs(
        video_path,
        attr_prefix="source_video",
        extra={
            "codec": metadata.get("codec"),
            "pix_fmt": metadata.get("pix_fmt"),
            "width": metadata.get("width"),
            "height": metadata.get("height"),
            "fps": metadata.get("fps"),
            "frame_count": metadata.get("total_frames"),
        },
    )
    return {
        "strategy": live["source_video_fingerprint_strategy"],
        "value": live["source_video_fingerprint"],
        "size_bytes": live["source_video_size_bytes"],
        "mtime_ns": live["source_video_mtime_ns"],
        "relocation_stable": False,
    }


def _authority_node_exists(root: zarr.Group, authority_path: str) -> bool:
    try:
        return isinstance(root.get(authority_path), zarr.Group)
    except Exception:
        return False


def _validate_published_authority(
    root: zarr.Group,
    *,
    camera_id: str,
    authority_path: str,
) -> dict[str, str]:
    status = load_acquisition_authority_publication_status(root)
    if (
        status.status != ACQUISITION_AUTHORITY_PUBLISHED
        or status.authority_mode != EXTERNAL_ACQUISITION_AUTHORITY_MODE
        or status.authority_path != authority_path
    ):
        raise PixelFrameAuthorityError(
            "Published acquisition status is not the expected external-video authority."
        )
    ownership, frame = load_persisted_acquisition_camera_authority(
        root,
        expected_camera_id=camera_id,
    )
    if ownership.record.mode != EXTERNAL_ACQUISITION_AUTHORITY_MODE:
        raise PixelFrameAuthorityError(
            "Persisted acquisition ownership is not external-video mode."
        )
    return {
        "authority_path": authority_path,
        "ownership_record_ref": ownership.record_ref,
        "ownership_record_sha256": ownership.record_sha256,
        "frame_record_ref": frame.record_ref,
        "frame_record_sha256": frame.record_sha256,
    }


def repair_external_video_acquisition_authority(
    zarr_path: str | Path,
    *,
    apply: bool = False,
) -> dict[str, Any]:
    """Audit one archive and optionally publish its missing authority."""

    archive_path = Path(zarr_path).expanduser().resolve()
    report: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "zarr_path": str(archive_path),
        "apply": bool(apply),
        "status": "blocked",
        "action": None,
        "source_video_path": None,
        "recording_id": None,
        "camera_id": None,
        "authority_path": None,
        "live_fingerprint_match": False,
        "legacy_imageio_metadata_present": False,
        "legacy_imageio_metadata_action": None,
        "error": None,
    }
    try:
        root = open_zarr_group_direct(archive_path, mode="a" if apply else "r")
        resolved = resolve_source_video(
            root,
            zarr_path=archive_path,
            require_exists=True,
        )
        raw_metadata = root.attrs.get("source_video_metadata")
        if not isinstance(raw_metadata, Mapping):
            raise PixelFrameAuthorityError(
                "Archive root is missing canonical source_video_metadata."
            )
        metadata_candidate = copy.deepcopy(dict(raw_metadata))
        nested_imageio_present = "imageio_metadata" in metadata_candidate
        metadata_candidate.pop("imageio_metadata", None)
        top_imageio_present = "imageio_metadata" in root.attrs
        legacy_imageio_present = nested_imageio_present or top_imageio_present
        metadata = parse_source_video_metadata(metadata_candidate)
        recording_id = root.attrs.get("recording_id")
        camera_id = root.attrs.get("camera_id")
        if type(recording_id) is not str or not recording_id.strip():
            raise PixelFrameAuthorityError(
                "Archive root is missing exact recording_id."
            )
        if type(camera_id) is not str or not camera_id.strip():
            raise PixelFrameAuthorityError("Archive root is missing exact camera_id.")
        if metadata.get("camera_id") != camera_id:
            raise PixelFrameAuthorityError(
                "source_video_metadata.camera_id conflicts with root camera_id."
            )
        expected_fingerprint = _expected_live_fingerprint(resolved.path, metadata)
        if metadata.get("file_fingerprint") != expected_fingerprint:
            raise PixelFrameAuthorityError(
                "Live source video differs from the persisted stat_v1 fingerprint."
            )

        authority_path = f"analysis/acquisition_camera_frames/{camera_id}"
        report.update(
            {
                "source_video_path": str(resolved.path),
                "recording_id": recording_id,
                "camera_id": camera_id,
                "authority_path": authority_path,
                "live_fingerprint_match": True,
                "legacy_imageio_metadata_present": legacy_imageio_present,
                "legacy_imageio_metadata_action": (
                    "would_remove" if legacy_imageio_present and not apply else None
                ),
            }
        )

        raw = root.get("raw_video")
        root_status_value = root.attrs.get(ACQUISITION_AUTHORITY_STATUS_ATTR)
        raw_status_value = (
            raw.attrs.get(ACQUISITION_AUTHORITY_STATUS_ATTR)
            if isinstance(raw, zarr.Group)
            else None
        )
        authority_exists = _authority_node_exists(root, authority_path)

        if root_status_value is None and raw_status_value is None:
            if authority_exists:
                raise PixelFrameAuthorityError(
                    "Statusless acquisition authority is ambiguous and requires manual repair."
                )
            action = "publish" if apply else "would_publish"
        else:
            if root_status_value is None or raw_status_value is None:
                raise PixelFrameAuthorityError(
                    "Root/raw acquisition publication status is incomplete."
                )
            status = load_acquisition_authority_publication_status(root)
            if (
                status.authority_mode != EXTERNAL_ACQUISITION_AUTHORITY_MODE
                or status.authority_path != authority_path
                or status.status
                not in {ACQUISITION_AUTHORITY_PENDING, ACQUISITION_AUTHORITY_PUBLISHED}
            ):
                raise PixelFrameAuthorityError(
                    "Existing acquisition publication status conflicts with this repair."
                )
            if status.status == ACQUISITION_AUTHORITY_PUBLISHED:
                evidence = _validate_published_authority(
                    root,
                    camera_id=camera_id,
                    authority_path=authority_path,
                )
                report.update(evidence)
                if legacy_imageio_present:
                    if not apply:
                        report.update(
                            {
                                "status": "ok",
                                "action": "would_remove_legacy_metadata",
                                "legacy_imageio_metadata_action": "would_remove",
                            }
                        )
                        return report
                    _remove_legacy_imageio_metadata(
                        root,
                        canonical_source_video_metadata=metadata,
                    )
                    consolidate_metadata_capture_expected_warnings(archive_path)
                    _validate_published_authority(
                        root,
                        camera_id=camera_id,
                        authority_path=authority_path,
                    )
                    report.update(
                        {
                            "status": "ok",
                            "action": "removed_legacy_metadata",
                            "legacy_imageio_metadata_action": "removed",
                            "consolidated_metadata_updated": True,
                        }
                    )
                    return report
                report.update({"status": "ok", "action": "already_complete"})
                return report
            action = "resume" if apply else "would_resume"

        if not apply:
            report.update({"status": "ok", "action": action})
            return report

        if legacy_imageio_present:
            if (
                root_status_value is not None
                or raw_status_value is not None
                or authority_exists
            ):
                raise PixelFrameAuthorityError(
                    "Legacy ImageIO metadata can only be removed before authority publication."
                )
            _remove_legacy_imageio_metadata(
                root,
                canonical_source_video_metadata=metadata,
            )
            report["legacy_imageio_metadata_action"] = "removed"

        publish_external_video_acquisition_authority(root)
        evidence = _validate_published_authority(
            root,
            camera_id=camera_id,
            authority_path=authority_path,
        )
        report.update(evidence)
        report.update({"status": "ok", "action": action})
        return report
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report


def _discover_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = [Path(value) for value in args.zarr_path]
    for list_path in args.path_list:
        paths.extend(load_path_list(Path(list_path)))
    if args.source == "filesystem":
        paths.extend(
            path
            for path in discover_filesystem_zarrs(
                args.scope,
                recursive=bool(args.recursive),
            )
            if path.name.endswith("_analysis.zarr")
        )
    elif args.source == "registry":
        registry_path = (
            (args.registry or RegistryPaths.from_env(Path.cwd()).path)
            .expanduser()
            .resolve()
        )
        paths.extend(
            discover_registry_zarrs(
                registry_path=registry_path,
                scope_paths=args.scope,
                zarr_use="analysis",
                path_contains=args.path_contains,
                zarr_suffix="_analysis.zarr",
            )
        )
    return sorted({path.expanduser().resolve() for path in paths})


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", nargs="*", type=Path)
    parser.add_argument("--path-list", action="append", default=[], type=Path)
    parser.add_argument("--scope", action="append", default=[], type=Path)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--source", choices=("none", "filesystem", "registry"), default="none"
    )
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--path-contains", type=str)
    parser.add_argument(
        "--apply", action="store_true", help="Publish repairs; default is dry-run."
    )
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    paths = _discover_paths(args)
    if not paths:
        parser.error("No analysis Zarr paths provided or discovered.")
    reports = [
        repair_external_video_acquisition_authority(path, apply=bool(args.apply))
        for path in paths
    ]
    blocked = sum(report["status"] != "ok" for report in reports)
    result = {
        "schema_id": SCHEMA_ID,
        "status": "failed" if blocked else "ok",
        "apply": bool(args.apply),
        "zarr_count": len(reports),
        "blocked_zarr_count": blocked,
        "would_publish_zarr_count": sum(
            report["action"] == "would_publish" for report in reports
        ),
        "published_zarr_count": sum(
            report["action"] == "publish" for report in reports
        ),
        "already_complete_zarr_count": sum(
            report["action"] == "already_complete" for report in reports
        ),
        "zarrs": reports,
    }
    if args.output_json is not None:
        write_json_atomic(args.output_json.expanduser().resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if blocked == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
