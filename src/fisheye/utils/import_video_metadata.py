#!/usr/bin/env python3
"""Write metadata-only raw_video attributes for production (no frame import)."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import zarr

from fisheye.shared.import_video_metadata import (
    _preview_updates,
    _probe_video,
    _write_metadata,
    probe_video_metadata,
    write_video_metadata,
)

__all__ = [
    "_preview_updates",
    "_probe_video",
    "_write_metadata",
    "main",
    "probe_video_metadata",
    "write_video_metadata",
]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_path", type=Path)
    parser.add_argument("zarr_path", type=Path)
    parser.add_argument("--apply", action="store_true", help="Write metadata to the zarr.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing attrs.")
    parser.add_argument("--import-purpose", default="production", help="Recorded import_purpose value.")

    args = parser.parse_args(argv)

    video_path = args.video_path.expanduser()
    zarr_path = args.zarr_path.expanduser()
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if not zarr_path.exists():
        raise FileNotFoundError(zarr_path)

    meta = _probe_video(video_path)

    root = zarr.open_group(str(zarr_path), mode="r+")

    raw = root.get("raw_video")
    has_arrays = False
    if raw is not None:
        has_arrays = any(name in raw for name in ("images_full", "images_ds", "images_ds_rgb"))
    if args.import_purpose == "training_data":
        zarr_purpose = "training"
    elif args.import_purpose == "production":
        zarr_purpose = "production"
    else:
        zarr_purpose = "analysis"

    raw_payload = {
        "import_method": "metadata_only",
        "import_mode": "metadata_only",
        "import_stage": "metadata_only",
        "import_timestamp": datetime.now(timezone.utc).isoformat(),
        "import_purpose": args.import_purpose,
        "fps": meta.get("fps"),
        "total_frames": meta.get("total_frames"),
        "source_video": meta.get("source_video"),
        "source_path": meta.get("source_path"),
        "original_resolution": (meta.get("height"), meta.get("width")),
        "video_codec": meta.get("codec"),
        "video_pix_fmt": meta.get("pix_fmt"),
        "has_full_resolution": has_arrays and "images_full" in raw,
        "has_downsampled": has_arrays and ("images_ds" in raw or "images_ds_rgb" in raw),
    }
    root_payload = {
        "has_raw_video": has_arrays,
        "zarr_purpose": zarr_purpose,
        "source_video": meta.get("source_video"),
        "source_video_path": meta.get("source_path"),
        "source_path": meta.get("source_path"),
        "width": meta.get("width"),
        "height": meta.get("height"),
        "fps": meta.get("fps"),
        "total_frames": meta.get("total_frames"),
        "n_frames": meta.get("total_frames"),
        "duration_seconds": meta.get("duration_seconds"),
        "video_codec": meta.get("codec"),
        "video_pix_fmt": meta.get("pix_fmt"),
        "source_video_metadata": meta,
    }

    if not args.apply:
        raw_attrs = raw.attrs if raw is not None else {}
        raw_updates = _preview_updates(raw_attrs, raw_payload, overwrite=args.overwrite)
        root_updates = _preview_updates(root.attrs, root_payload, overwrite=args.overwrite)
        print("DRY RUN: metadata-only import (no changes written)")
        print(f"  zarr: {zarr_path}")
        print(f"  video: {video_path}")
        print(f"  root updates: {sorted(root_updates.keys())}")
        print(f"  raw_video updates: {sorted(raw_updates.keys())}")
        if raw_updates:
            print("  raw_video changes:")
            for key in sorted(raw_updates.keys()):
                current = raw_attrs.get(key) if raw is not None else None
                print(f"    {key}: {current} -> {raw_updates[key]}")
        if root_updates:
            print("  root changes:")
            for key in sorted(root_updates.keys()):
                current = root.attrs.get(key)
                print(f"    {key}: {current} -> {root_updates[key]}")
        return 0

    updates = _write_metadata(root, meta, overwrite=args.overwrite, import_purpose=args.import_purpose)
    print(f"Wrote metadata-only raw_video attrs to {zarr_path}")
    print(f"  root attrs updated: {len(updates['root'])}")
    print(f"  raw_video attrs updated: {len(updates['raw_video'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
