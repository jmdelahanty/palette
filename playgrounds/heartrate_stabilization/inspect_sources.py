from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from _common import (
    cfg_path,
    cfg_value,
    crop_row_frame_id,
    get_video_info,
    load_config,
    read_crop_meta,
    zarr_json_metadata,
)


def _shape_text(metadata: Mapping[str, Any]) -> str:
    shape = metadata.get("shape")
    dtype = metadata.get("data_type")
    chunks = metadata.get("chunk_grid", {}).get("configuration", {}).get("chunk_shape")
    return f"shape={shape} dtype={dtype} chunks={chunks}"


def _mask_parent_summary(zarr_path: Path, parent_name: str) -> None:
    parent_path = zarr_path / parent_name
    if not parent_path.exists():
        print(f"mask_parent: {parent_name}: missing")
        return
    run_names = sorted(path.name for path in parent_path.iterdir() if path.is_dir() and (path / "zarr.json").exists())
    print(f"mask_parent: {parent_name}: runs={run_names or []}")
    for run_name in run_names[-3:]:
        run_path = f"{parent_name}/{run_name}"
        try:
            run_metadata = zarr_json_metadata(zarr_path, run_path)
        except FileNotFoundError:
            print(f"mask_run: {run_path}: missing_metadata")
            continue
        attrs = run_metadata.get("attributes", {})
        print(
            f"mask_run: {run_path}: "
            f"source_crop_run={attrs.get('source_crop_run')} "
            f"mask_labels={attrs.get('mask_labels')}"
        )
        for array_name in ("frame_indices", "source_crop_row_ids", "masks_roi", "mask_probs_roi"):
            try:
                metadata = zarr_json_metadata(zarr_path, f"{run_path}/{array_name}")
            except FileNotFoundError:
                continue
            print(f"mask_array: {run_path}/{array_name}: {_shape_text(metadata)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect heartrate stabilization playground inputs.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.example.toml"),
        help="Playground TOML config.",
    )
    parser.add_argument(
        "--read-zarr-slice",
        action="store_true",
        help="Open zarr with Python and print a tiny slice. Avoid this in sandboxes where /groups zarr reads hang.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    crop_video = cfg_path(config, "inputs", "crop_video")
    crop_meta_csv = cfg_path(config, "inputs", "crop_meta_csv")
    zarr_path = cfg_path(config, "inputs", "zarr_path")
    keypoint_group = str(cfg_value(config, "inputs", "keypoint_group"))
    frame_id_column = str(cfg_value(config, "alignment", "frame_id_column", "camera_frame_id"))

    print(f"config: {args.config}")
    print(f"crop_video: {crop_video}")
    print(f"crop_meta_csv: {crop_meta_csv}")
    print(f"zarr_path: {zarr_path}")
    print(f"keypoint_group: {keypoint_group}")

    video = get_video_info(crop_video)
    print(f"crop_video_info: width={video.width} height={video.height} frames={video.frame_count} fps={video.fps:g}")

    rows = read_crop_meta(crop_meta_csv)
    print(f"crop_meta_rows: {len(rows)}")
    if rows:
        first = rows[0]
        last = rows[-1]
        print(
            "crop_meta_frame_range: "
            f"{frame_id_column}={crop_row_frame_id(0, first, frame_id_column)}.."
            f"{crop_row_frame_id(len(rows) - 1, last, frame_id_column)} "
            f"recording_frame_id={first.get('recording_frame_id')}..{last.get('recording_frame_id')}"
        )
        print(
            "first_crop: "
            f"x={first.get('crop_x')} y={first.get('crop_y')} w={first.get('crop_w')} h={first.get('crop_h')} "
            f"detection_confidence={first.get('detection_confidence')}"
        )

    for array_name in (
        "frame_indices",
        "keypoints_img",
        "keypoints_roi",
        "usable_keypoints",
        "refined_success",
        "heading",
    ):
        try:
            metadata = zarr_json_metadata(zarr_path, f"{keypoint_group}/{array_name}")
        except FileNotFoundError:
            print(f"zarr_array: {array_name}: missing")
            continue
        print(f"zarr_array: {array_name}: {_shape_text(metadata)}")

    for mask_parent in ("refined_subject_masks_runs", "subject_mask_runs"):
        _mask_parent_summary(zarr_path, mask_parent)

    if args.read_zarr_slice:
        import zarr

        root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
        group = root[keypoint_group]
        print("zarr_slice_frame_indices:", np.asarray(group["frame_indices"][:5]).tolist())
        print("zarr_slice_keypoints_img_0:", np.asarray(group["keypoints_img"][:1]).round(2).tolist())


if __name__ == "__main__":
    main()
