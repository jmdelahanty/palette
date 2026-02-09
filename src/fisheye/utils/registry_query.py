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


def _query_dataset_ids_by_subject_lineage(
    registry: Registry,
    *,
    cross_id: Optional[str],
    genotype: Optional[str],
    dpf: Optional[int],
    dpf_min: Optional[int],
    dpf_max: Optional[int],
) -> set[str]:
    sql = [
        "SELECT DISTINCT d.dataset_id",
        "FROM datasets d",
        "JOIN recording_subject_overview rso ON rso.recording_id = d.recording_id",
        "WHERE 1=1",
    ]
    params: list[object] = []
    if cross_id is not None:
        sql.append("AND rso.cross_id = ?")
        params.append(str(cross_id))
    if genotype is not None:
        sql.append("AND rso.genotype = ?")
        params.append(str(genotype))
    if dpf is not None:
        sql.append("AND rso.dpf_at_acquisition = ?")
        params.append(int(dpf))
    if dpf_min is not None:
        sql.append("AND rso.dpf_at_acquisition >= ?")
        params.append(int(dpf_min))
    if dpf_max is not None:
        sql.append("AND rso.dpf_at_acquisition <= ?")
        params.append(int(dpf_max))
    rows = registry.conn.execute(" ".join(sql), params).fetchall()
    return {
        str(row["dataset_id"])
        for row in rows
        if row["dataset_id"] is not None
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, help="Optional registry SQLite path.")
    parser.add_argument("--dish-design", type=str, help="Exact dish design match.")
    parser.add_argument("--dish-design-like", type=str, help="Substring match for dish design.")
    parser.add_argument("--fish-id", type=str, help="Exact fish_id match.")
    parser.add_argument("--subject-count-min", type=int)
    parser.add_argument("--subject-count-max", type=int)
    parser.add_argument("--zarr-use", type=str, help="Exact zarr use match (training/analysis/inference/export/archive).")
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
    parser.add_argument("--cross-id", type=str, help="Exact cross_id match via recording subject lineage.")
    parser.add_argument("--genotype", type=str, help="Exact genotype match via recording subject lineage.")
    parser.add_argument("--dpf", type=int, help="Exact dpf_at_acquisition match via recording subject lineage.")
    parser.add_argument("--dpf-min", type=int, help="Minimum dpf_at_acquisition via recording subject lineage.")
    parser.add_argument("--dpf-max", type=int, help="Maximum dpf_at_acquisition via recording subject lineage.")
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
    if (
        args.dpf_min is not None
        and args.dpf_max is not None
        and int(args.dpf_min) > int(args.dpf_max)
    ):
        raise SystemExit("--dpf-min must be <= --dpf-max.")

    registry_path = args.registry or RegistryPaths.from_env(Path.cwd()).path
    registry = Registry(registry_path)
    use_subject_filters = any(
        value is not None
        for value in (args.cross_id, args.genotype, args.dpf, args.dpf_min, args.dpf_max)
    )
    rows = registry.query_datasets(
        dish_design=args.dish_design,
        dish_design_like=args.dish_design_like,
        fish_id=args.fish_id,
        subject_count_min=args.subject_count_min,
        subject_count_max=args.subject_count_max,
        zarr_use=args.zarr_use,
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
        # Apply row limit after subject-lineage filtering to avoid dropping
        # candidate rows before cross/genotype/DPF constraints are evaluated.
        limit=(None if use_subject_filters else args.limit),
    )
    if use_subject_filters:
        try:
            allowed_dataset_ids = _query_dataset_ids_by_subject_lineage(
                registry,
                cross_id=args.cross_id,
                genotype=args.genotype,
                dpf=args.dpf,
                dpf_min=args.dpf_min,
                dpf_max=args.dpf_max,
            )
        except Exception as exc:
            registry.close()
            raise SystemExit(
                "Subject-lineage filters (--cross-id/--genotype/--dpf/--dpf-min/--dpf-max) require "
                f"`recording_subject_overview` to be queryable: {exc}"
            ) from exc
        rows = [
            row
            for row in rows
            if str(row["dataset_id"]) in allowed_dataset_ids
        ]
        if args.limit is not None:
            rows = rows[: int(args.limit)]
    registry.close()

    if args.json:
        payload = [dict(row) for row in rows]
        print(json.dumps(payload, indent=2))
    else:
        for row in rows:
            encoder_name = row["encoder_name"] if "encoder_name" in row.keys() else None
            format_encoder = row["format_encoder"] if "format_encoder" in row.keys() else None
            print(
                f"{row['zarr_path']}\t"
                f"dish={row['dish_design'] or '-'}\t"
                f"fps={row['fps'] or '-'}\t"
                f"exposure_us={row['exposure'] or '-'}\t"
                f"codec={row['video_codec'] or '-'}\t"
                f"pixfmt={row['video_pix_fmt'] or '-'}\t"
                f"encoder={encoder_name or format_encoder or '-'}"
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
