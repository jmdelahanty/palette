#!/usr/bin/env python3
"""Query the registry for datasets matching acquisition/provenance filters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from fisheye.registry.db import Registry, RegistryPaths


def _as_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, help="Optional registry SQLite path.")
    parser.add_argument("--dish-design", type=str, help="Exact dish design match.")
    parser.add_argument("--dish-design-like", type=str, help="Substring match for dish design.")
    parser.add_argument("--fish-id", type=str, help="Exact fish_id match.")
    parser.add_argument("--subject-count-min", type=int)
    parser.add_argument("--subject-count-max", type=int)
    parser.add_argument("--zarr-purpose", type=str, help="Exact zarr purpose match (training/production/analysis).")
    parser.add_argument("--fps-min", type=float)
    parser.add_argument("--fps-max", type=float)
    parser.add_argument("--exposure-min", type=float)
    parser.add_argument("--exposure-max", type=float)
    parser.add_argument("--frame-rate-min", type=float)
    parser.add_argument("--frame-rate-max", type=float)
    parser.add_argument("--gain-min", type=float)
    parser.add_argument("--gain-max", type=float)
    parser.add_argument("--video-codec", type=str)
    parser.add_argument("--video-pix-fmt", type=str)
    parser.add_argument("--format-encoder", type=str, help="Exact match on container encoder tag.")
    parser.add_argument("--format-title", type=str, help="Exact match on container title tag.")
    parser.add_argument("--format-comment", type=str, help="Exact match on container comment tag.")
    parser.add_argument("--encoder-name", type=str, help="Exact match on encoder name in comment.")
    parser.add_argument("--encoder-codec", type=str, help="Exact match on encoder codec in comment.")
    parser.add_argument("--encoder-preset", type=str, help="Exact match on encoder preset in comment.")
    parser.add_argument("--encoder-tuning", type=str, help="Exact match on encoder tuning in comment.")
    parser.add_argument("--encoder-rc", type=str, help="Exact match on encoder rate control in comment.")
    parser.add_argument("--compression", type=str, help="Compression name (e.g., lz4, zstd).")
    parser.add_argument("--camera-model", type=str)
    parser.add_argument("--camera-serial", type=str)
    parser.add_argument("--camera-id", type=str)
    parser.add_argument("--rig-id", type=str)
    parser.add_argument("--arena-id", type=str)
    parser.add_argument(
        "--model-input",
        choices=["gray", "rgb"],
        help="Filter datasets by available downsample modality required for training.",
    )
    parser.add_argument("--path-contains", type=str)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    parser.add_argument("--output-file-list", type=Path, help="Write matched zarr paths to file.")

    args = parser.parse_args(argv)

    registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
    registry = Registry(registry_path)
    rows = registry.query_datasets(
        dish_design=args.dish_design,
        dish_design_like=args.dish_design_like,
        fish_id=args.fish_id,
        subject_count_min=args.subject_count_min,
        subject_count_max=args.subject_count_max,
        zarr_purpose=args.zarr_purpose,
        fps_min=args.fps_min,
        fps_max=args.fps_max,
        exposure_min=args.exposure_min,
        exposure_max=args.exposure_max,
        frame_rate_min=args.frame_rate_min,
        frame_rate_max=args.frame_rate_max,
        gain_min=args.gain_min,
        gain_max=args.gain_max,
        video_codec=args.video_codec,
        video_pix_fmt=args.video_pix_fmt,
        format_encoder=args.format_encoder,
        format_title=args.format_title,
        format_comment=args.format_comment,
        encoder_name=args.encoder_name,
        encoder_codec=args.encoder_codec,
        encoder_preset=args.encoder_preset,
        encoder_tuning=args.encoder_tuning,
        encoder_rc=args.encoder_rc,
        compression_name=args.compression,
        camera_model=args.camera_model,
        camera_serial=args.camera_serial,
        camera_id=args.camera_id,
        rig_id=args.rig_id,
        arena_id=args.arena_id,
        model_input=args.model_input,
        path_contains=args.path_contains,
        limit=args.limit,
    )
    registry.close()

    if args.json:
        payload = [dict(row) for row in rows]
        print(json.dumps(payload, indent=2))
    else:
        for row in rows:
            print(
                f"{row['zarr_path']}\t"
                f"dish={row['dish_design'] or '-'}\t"
                f"fps={row['fps'] or '-'}\t"
                f"exposure_us={row['exposure'] or '-'}\t"
                f"codec={row['video_codec'] or '-'}\t"
                f"pixfmt={row['video_pix_fmt'] or '-'}\t"
                f"encoder={row.get('encoder_name') or row.get('format_encoder') or '-'}"
            )

    if args.output_file_list:
        args.output_file_list.parent.mkdir(parents=True, exist_ok=True)
        args.output_file_list.write_text(
            "\n".join([row["zarr_path"] for row in rows]) + ("\n" if rows else ""),
            encoding="utf-8",
        )
        print(f"Wrote {len(rows)} paths to {args.output_file_list}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
