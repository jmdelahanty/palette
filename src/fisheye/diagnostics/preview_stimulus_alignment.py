#!/usr/bin/env python3
"""
Quick visual sanity check that the raw camera video and the exported stimulus video match.

This script displays (or saves) a side-by-side comparison of:
  * the first frame from the stimulus-generated video (often an H.264 MP4), and
  * the corresponding frame from the raw CamXXXX video, selected via an explicit frame
    index or deduced from an associated HDF5 file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _import_optional_h5py():
    try:
        import h5py  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("h5py is required for --h5 lookup but is not installed.") from exc
    return h5py


def _load_frame_range(
    video_path: Path,
    start_index: int,
    count: int,
    *,
    prefer_gpu: bool = False,
    quiet: bool = False,
) -> Tuple[list[np.ndarray], np.ndarray]:
    try:
        from decord import VideoReader, cpu, gpu  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "decord is required for fast frame access. Install with `pip install decord`."
        ) from exc

    try:
        ctx = gpu(0) if prefer_gpu else cpu(0)
    except Exception:
        ctx = cpu(0)

    try:
        vr = VideoReader(str(video_path), ctx=ctx)
    except Exception as exc:
        if prefer_gpu:
            if not quiet:
                print(f"[preview] GPU decode failed ({exc}); retrying on CPU.")
            vr = VideoReader(str(video_path), ctx=cpu(0))
        else:
            raise

    total_frames = len(vr)
    if start_index < 0 or start_index >= total_frames:
        raise ValueError(
            f"Start index {start_index} out of range for {video_path} (len={total_frames})."
        )

    end_index = min(start_index + max(count, 1), total_frames)
    frames: list[np.ndarray] = []
    frame_indices = np.arange(start_index, end_index, dtype=np.int64)

    for idx in frame_indices:
        frame = vr[idx].asnumpy()
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        frames.append(frame)

    return frames, frame_indices


DEFAULT_H5_DATASETS = (
    ("video_metadata/frame_metadata", "triggering_camera_frame_id"),
    ("stimulus/camera_frame", None),
    ("stimulus/frame_camera_index", None),
    ("stimulus/video_frame_index", None),
    ("video/camera_frame", None),
    ("camera_frame", None),
)


def _parse_dataset_spec(spec: str) -> Tuple[str, Optional[str]]:
    if ":" in spec:
        path, field = spec.split(":", 1)
        return path.strip(), field.strip() or None
    return spec.strip(), None


def _lookup_camera_frame_from_h5(
    h5_path: Path, dataset_candidates: Iterable[Tuple[str, Optional[str]]]
) -> Tuple[int, str]:
    h5py = _import_optional_h5py()
    with h5py.File(h5_path, "r") as handle:
        for path, field in dataset_candidates:
            if path not in handle:
                continue

            dataset = handle[path]
            if dataset.size == 0:
                continue

            value = dataset[()] if dataset.shape == () else dataset[0]

            if isinstance(value, np.void) and dataset.dtype.names:
                target_field = field
                if target_field and target_field in dataset.dtype.names:
                    frame_idx = int(value[target_field])
                else:
                    extracted: Optional[int] = None
                    for name in dataset.dtype.names:
                        if np.issubdtype(dataset.dtype[name], np.integer):
                            extracted = int(value[name])
                            break
                    if extracted is None:
                        continue
                    frame_idx = extracted
            elif isinstance(value, np.ndarray):
                if value.size == 0:
                    continue
                frame_idx = int(np.asarray(value).astype(np.int64).flat[0])
            else:
                frame_idx = int(value)

            field_suffix = f":{field}" if field else ""
            return frame_idx, f"{path}{field_suffix}"

    pretty = [f"{path}:{field}" if field else path for path, field in dataset_candidates]
    raise RuntimeError(
        f"None of the expected datasets {pretty} were found in {h5_path}. "
        "Provide --camera-frame explicitly or pass --h5-dataset."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize a raw camera frame alongside the first stimulus frame."
    )
    parser.add_argument(
        "--camera-video",
        required=True,
        type=Path,
        help="Path to the raw Cam*.mp4 video.",
    )
    parser.add_argument(
        "--stimulus-video",
        required=True,
        type=Path,
        help="Path to the stimulus-generated video (H.264 MP4).",
    )
    parser.add_argument(
        "--camera-frame",
        type=int,
        help="Explicit camera frame index to load from the raw video.",
    )
    parser.add_argument(
        "--h5",
        type=Path,
        help="Optional HDF5 file containing the camera_frame index.",
    )
    parser.add_argument(
        "--h5-dataset",
        type=str,
        help="Optional dataset path within the HDF5 file. You can specify 'path:field' to pick a particular field in a structured dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the comparison figure instead of showing interactively.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=120,
        help="Number of consecutive frames to decode for navigation (default: 120).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational logging.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    camera_video = args.camera_video.expanduser()
    stimulus_video = args.stimulus_video.expanduser()

    if not camera_video.exists():
        raise SystemExit(f"Camera video not found: {camera_video}")
    if not stimulus_video.exists():
        raise SystemExit(f"Stimulus video not found: {stimulus_video}")

    chosen_frame = args.camera_frame
    source_desc = "provided via --camera-frame"

    if chosen_frame is None and args.h5:
        h5_path = args.h5.expanduser()
        if not h5_path.exists():
            raise SystemExit(f"HDF5 file not found: {h5_path}")
        if args.h5_dataset:
            datasets = [_parse_dataset_spec(args.h5_dataset)]
        else:
            datasets = DEFAULT_H5_DATASETS
        chosen_frame, ds_used = _lookup_camera_frame_from_h5(h5_path, datasets)
        source_desc = f"from {h5_path} dataset '{ds_used}'"

    if chosen_frame is None:
        chosen_frame = 0
        source_desc = "defaulted to frame 0"

    if not args.quiet:
        print(f"Camera video : {camera_video}")
        print(f"Stimulus video: {stimulus_video}")
        print(f"Camera frame  : {chosen_frame} ({source_desc})")

    chunk_size = max(1, args.chunk_size)
    camera_frames, camera_indices = _load_frame_range(
        camera_video,
        chosen_frame,
        chunk_size,
        prefer_gpu=True,
        quiet=args.quiet,
    )
    stimulus_frames, stimulus_indices = _load_frame_range(
        stimulus_video,
        0,
        chunk_size,
        prefer_gpu=False,
        quiet=args.quiet,
    )

    total_pairs = min(len(camera_frames), len(stimulus_frames))
    if total_pairs == 0:
        raise SystemExit("No frames available for comparison in the requested range.")

    if not args.quiet and total_pairs < chunk_size:
        print(f"[preview] Only {total_pairs} paired frames available (chunk truncated).")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax_cam, ax_stim = axes
    img_cam = ax_cam.imshow(camera_frames[0])
    img_stim = ax_stim.imshow(stimulus_frames[0])
    ax_cam.axis("off")
    ax_stim.axis("off")

    fig.suptitle("Camera ↔ Stimulus Alignment Check (←/→ to navigate)", fontsize=15)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    state = {"index": 0}

    def update_display(new_index: int) -> None:
        state["index"] = new_index
        img_cam.set_data(camera_frames[new_index])
        img_stim.set_data(stimulus_frames[new_index])
        ax_cam.set_title(f"Camera frame {camera_indices[new_index]}")
        ax_stim.set_title(f"Stimulus frame {stimulus_indices[new_index]}")
        fig.canvas.draw_idle()

    if args.output:
        output_path = args.output.expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        update_display(0)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        if not args.quiet:
            print(f"Saved comparison figure to {output_path}")
        plt.close(fig)
    else:
        if not args.quiet:
            print("Use left/right arrow keys to move through the frame chunk.")

        def on_key(event):
            if event.key in ("right", "d"):
                if state["index"] < total_pairs - 1:
                    update_display(state["index"] + 1)
            elif event.key in ("left", "a"):
                if state["index"] > 0:
                    update_display(state["index"] - 1)

        fig.canvas.mpl_connect("key_press_event", on_key)
        update_display(0)
        plt.show()

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
