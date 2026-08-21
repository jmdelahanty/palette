#!/usr/bin/env python3
"""Audit or repair acquisition authority for clipped analysis Zarrs.

The command is dry-run by default.  It validates the complete parent frame
index, probes every clip's encoded extent, and compares live member fingerprints
before writing any authority metadata.  It does not copy or decode video pixels.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Sequence

import zarr

from fisheye.shared.acquisition_publication_status import (
    ACQUISITION_AUTHORITY_PENDING,
    ACQUISITION_AUTHORITY_PUBLISHED,
    ACQUISITION_AUTHORITY_STATUS_ATTR,
    CLIPPED_EXTERNAL_ACQUISITION_AUTHORITY_MODE,
    load_acquisition_authority_publication_status,
)
from fisheye.shared.clipped_video_collection import (
    build_clipped_video_collection_metadata,
    clipped_video_collection_summary,
)
from fisheye.shared.import_video_metadata import (
    publish_clipped_video_collection_acquisition_authority,
)
from fisheye.shared.pixel_frame_authority import (
    load_persisted_acquisition_camera_authority,
)
from fisheye.shared.zarr_helpers import open_zarr_group_direct


SCHEMA_ID = "palette.clipped_analysis_acquisition_authority_repair.v1"


def _authority_exists(root: zarr.Group, camera_id: str) -> bool:
    try:
        return isinstance(
            root.get(f"analysis/acquisition_camera_frames/{camera_id}"), zarr.Group
        )
    except Exception:
        return False


def repair_clipped_analysis_acquisition_authority(
    zarr_path: str | Path,
    *,
    apply: bool = False,
) -> dict[str, Any]:
    """Audit one clipped archive and optionally publish its missing authority."""

    archive = Path(zarr_path).expanduser().resolve()
    report: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "zarr_path": str(archive),
        "apply": bool(apply),
        "status": "blocked",
        "action": None,
        "recording_path": None,
        "recording_id": None,
        "camera_id": None,
        "acquisition_source": None,
        "authority_path": None,
        "error": None,
    }
    try:
        root = open_zarr_group_direct(archive, mode="a" if apply else "r")
        if root.attrs.get("analysis_layout") != "clipped_recording_shell":
            raise ValueError("Archive is not a clipped recording analysis shell.")
        inferred_recording = archive.parent.parent.resolve()
        declared_recording = (
            Path(str(root.attrs.get("recording_path") or inferred_recording))
            .expanduser()
            .resolve()
        )
        if declared_recording != inferred_recording:
            raise ValueError(
                "Archive recording_path differs from its <recording>/zarr location."
            )
        recording_id = str(root.attrs.get("recording_id") or "").strip()
        if not recording_id or recording_id != inferred_recording.name:
            raise ValueError(
                "Archive recording_id differs from its recording directory."
            )
        metadata = build_clipped_video_collection_metadata(
            inferred_recording,
            clip_index_path=root.attrs.get("recording_clip_index_json"),
            frame_index_path=root.attrs.get("recording_frame_index_path"),
            frame_manifest_path=root.attrs.get("recording_frame_index_manifest_path"),
        )
        camera_id = str(metadata["camera_id"])
        authority_path = f"analysis/acquisition_camera_frames/{camera_id}"
        report.update(
            {
                "recording_path": str(inferred_recording),
                "recording_id": recording_id,
                "camera_id": camera_id,
                "acquisition_source": clipped_video_collection_summary(metadata),
                "authority_path": authority_path,
            }
        )
        existing_camera = root.attrs.get("camera_id")
        if existing_camera not in (None, camera_id):
            raise ValueError(
                "Existing camera_id conflicts with clipped source evidence."
            )
        existing_metadata = root.attrs.get("source_video_metadata")
        if existing_metadata is not None and existing_metadata != metadata:
            raise ValueError(
                "Existing source_video_metadata conflicts with live clipped source evidence."
            )
        raw = root.get("raw_video")
        if not isinstance(raw, zarr.Group):
            raise ValueError("Archive lacks raw_video group.")
        if raw.attrs.get("storage_mode") != "external_clips" or "images_full" in raw:
            raise ValueError("Archive is not an external-clips metadata-only source.")
        root_status = root.attrs.get(ACQUISITION_AUTHORITY_STATUS_ATTR)
        raw_status = raw.attrs.get(ACQUISITION_AUTHORITY_STATUS_ATTR)
        authority_exists = _authority_exists(root, camera_id)
        if root_status is None and raw_status is None:
            if authority_exists:
                raise ValueError(
                    "Statusless clipped acquisition authority is ambiguous and requires manual repair."
                )
            action = "publish" if apply else "would_publish"
        else:
            if root_status is None or raw_status is None:
                raise ValueError(
                    "Root/raw acquisition publication status is incomplete."
                )
            status = load_acquisition_authority_publication_status(root)
            if (
                status.authority_mode != CLIPPED_EXTERNAL_ACQUISITION_AUTHORITY_MODE
                or status.authority_path != authority_path
                or status.status
                not in {ACQUISITION_AUTHORITY_PENDING, ACQUISITION_AUTHORITY_PUBLISHED}
            ):
                raise ValueError(
                    "Existing acquisition publication status conflicts with clipped authority."
                )
            if status.status == ACQUISITION_AUTHORITY_PUBLISHED:
                if existing_metadata != metadata or existing_camera != camera_id:
                    raise ValueError(
                        "Published clipped authority lacks its exact source metadata mirrors."
                    )
                ownership, frame = load_persisted_acquisition_camera_authority(
                    root, expected_camera_id=camera_id
                )
                if ownership.record.mode != CLIPPED_EXTERNAL_ACQUISITION_AUTHORITY_MODE:
                    raise ValueError(
                        "Persisted authority is not clipped-collection mode."
                    )
                report.update(
                    {
                        "status": "ok",
                        "action": "already_complete",
                        "ownership_record_ref": ownership.record_ref,
                        "ownership_record_sha256": ownership.record_sha256,
                        "frame_record_ref": frame.record_ref,
                        "frame_record_sha256": frame.record_sha256,
                    }
                )
                return report
            action = "resume" if apply else "would_resume"

        if not apply:
            report.update({"status": "ok", "action": action})
            return report

        attrs = copy.deepcopy(dict(root.attrs))
        attrs["camera_id"] = camera_id
        attrs["source_video_metadata"] = metadata
        root.attrs.put(attrs)
        if (
            root.attrs.get("camera_id") != camera_id
            or root.attrs.get("source_video_metadata") != metadata
        ):
            raise RuntimeError("Clipped source metadata did not round-trip exactly.")
        publication = publish_clipped_video_collection_acquisition_authority(root)
        ownership, frame = load_persisted_acquisition_camera_authority(
            root, expected_camera_id=camera_id
        )
        report.update(publication)
        report.update(
            {
                "status": "ok",
                "action": action,
                "ownership_record_ref": ownership.record_ref,
                "ownership_record_sha256": ownership.record_sha256,
                "frame_record_ref": frame.record_ref,
                "frame_record_sha256": frame.record_sha256,
            }
        )
        return report
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_paths", nargs="+", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    reports = [
        repair_clipped_analysis_acquisition_authority(path, apply=bool(args.apply))
        for path in args.zarr_paths
    ]
    blocked = sum(report["status"] != "ok" for report in reports)
    result = {
        "schema_id": SCHEMA_ID,
        "status": "failed" if blocked else "ok",
        "apply": bool(args.apply),
        "zarr_count": len(reports),
        "blocked_zarr_count": blocked,
        "zarrs": reports,
    }
    if args.output_json is not None:
        from fisheye.shared.json_safety import write_json_atomic

        write_json_atomic(args.output_json.expanduser().resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if blocked == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
