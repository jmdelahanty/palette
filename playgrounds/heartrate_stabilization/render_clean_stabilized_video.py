from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import subprocess
from typing import Protocol

import numpy as np

from _common import (
    cfg_path,
    cfg_value,
    compute_body_transform,
    crop_row_frame_id,
    ensure_output_dir,
    get_video_info,
    keypoints_to_crop_pixels,
    load_config,
    load_keypoint_data,
    read_crop_meta,
    selected_crop_rows,
)
from make_subset_clip import _apply_circular_mask


class VideoSink(Protocol):
    def write(self, frame_bgr: np.ndarray) -> None:
        ...

    def close(self) -> None:
        ...


class OpenCvVideoSink:
    def __init__(self, path: Path, *, fps: float, width: int, height: int) -> None:
        import cv2

        self.path = Path(path)
        self.writer = cv2.VideoWriter(
            str(self.path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps),
            (int(width), int(height)),
        )
        if not self.writer.isOpened():
            raise ValueError(f"Could not open output writer: {self.path}")

    def write(self, frame_bgr: np.ndarray) -> None:
        self.writer.write(frame_bgr[:, :, :3])

    def close(self) -> None:
        self.writer.release()


class FfmpegFfv1VideoSink:
    def __init__(
        self,
        path: Path,
        *,
        fps: float,
        width: int,
        height: int,
        ffmpeg_path: str = "ffmpeg",
    ) -> None:
        self.path = Path(path)
        self.width = int(width)
        self.height = int(height)
        self.process = subprocess.Popen(
            [
                str(ffmpeg_path),
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostats",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr0",
                "-s:v",
                f"{self.width}x{self.height}",
                "-r",
                f"{float(fps):.12g}",
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "ffv1",
                "-level",
                "3",
                "-g",
                "1",
                "-slicecrc",
                "1",
                "-pix_fmt",
                "bgr0",
                str(self.path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame_bgr: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin is closed")
        frame = np.asarray(frame_bgr)
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError(f"frame shape {frame.shape[:2]} does not match {(self.height, self.width)}")
        if frame.ndim == 2:
            bgr = np.repeat(frame[:, :, None], 3, axis=2)
        else:
            bgr = frame[:, :, :3]
        bgr = np.ascontiguousarray(bgr, dtype=np.uint8)
        bgr0 = np.empty((self.height, self.width, 4), dtype=np.uint8)
        bgr0[:, :, :3] = bgr
        bgr0[:, :, 3] = 0
        self.process.stdin.write(bgr0.tobytes())

    def close(self) -> None:
        stderr = b""
        if self.process.stdin is not None:
            self.process.stdin.close()
        if self.process.stderr is not None:
            stderr = self.process.stderr.read()
        return_code = self.process.wait()
        if return_code != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {detail}")


def _read_frame(capture, *, frame_index: int, next_expected_index: int | None):
    import cv2

    if next_expected_index is None or int(frame_index) != int(next_expected_index):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = capture.read()
    return ok, frame, int(frame_index) + 1


def _interpolation_flag(name: str) -> int:
    import cv2

    normalized = str(name).strip().lower()
    if normalized == "nearest":
        return cv2.INTER_NEAREST
    if normalized == "linear":
        return cv2.INTER_LINEAR
    if normalized == "cubic":
        return cv2.INTER_CUBIC
    raise ValueError("interpolation must be one of: nearest, linear, cubic")


def main() -> None:
    import cv2

    parser = argparse.ArgumentParser(description="Render a clean stabilized/rotated crop video.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.example.toml"))
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-count", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output clean video path. Defaults to [probe].output_dir/stabilized_clean_lossless.mkv.",
    )
    parser.add_argument(
        "--status-output",
        type=Path,
        default=None,
        help="Optional per-output-frame CSV with validity and transform metadata.",
    )
    parser.add_argument("--no-circular-mask", action="store_true", help="Do not black out stabilized warp corners.")
    parser.add_argument(
        "--codec",
        choices=("ffv1", "mp4v"),
        default="ffv1",
        help="Video encoding. ffv1 writes lossless Matroska; mp4v is lossy and for quick previews only.",
    )
    parser.add_argument("--ffmpeg-path", type=str, default="ffmpeg", help="FFmpeg executable for --codec ffv1.")
    parser.add_argument(
        "--interpolation",
        choices=("linear", "nearest", "cubic"),
        default="linear",
        help=(
            "Warp interpolation. Encoding can be lossless, but rotated pixels are still resampled. "
            "Use nearest only when exact source-pixel values matter more than visual quality."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    crop_video = cfg_path(config, "inputs", "crop_video")
    crop_meta_csv = cfg_path(config, "inputs", "crop_meta_csv")
    zarr_path = cfg_path(config, "inputs", "zarr_path")
    keypoint_group = str(cfg_value(config, "inputs", "keypoint_group"))
    frame_id_column = str(cfg_value(config, "alignment", "frame_id_column", "camera_frame_id"))
    frame_array = str(cfg_value(config, "alignment", "keypoint_frame_array", "frame_indices"))
    keypoint_array = str(cfg_value(config, "alignment", "keypoint_coordinate_array", "keypoints_img"))
    valid_array = str(cfg_value(config, "alignment", "keypoint_valid_array", "usable_keypoints"))
    stable_width = int(cfg_value(config, "alignment", "stable_width", 256))
    stable_height = int(cfg_value(config, "alignment", "stable_height", 256))
    stable_center_x = float(cfg_value(config, "alignment", "stable_center_x", stable_width / 2.0))
    stable_center_y = float(cfg_value(config, "alignment", "stable_center_y", stable_height / 2.0))
    origin = str(cfg_value(config, "alignment", "origin", "eye_midpoint"))
    target_forward = str(cfg_value(config, "alignment", "target_forward", "up"))
    scale = float(cfg_value(config, "alignment", "scale", 1.0))
    min_forward = float(cfg_value(config, "alignment", "min_forward_length_px", 8.0))
    min_eye_span = float(cfg_value(config, "alignment", "min_eye_span_px", 4.0))
    stable_circular_mask = bool(cfg_value(config, "alignment", "stable_circular_mask", True)) and not args.no_circular_mask
    stable_mask_radius = float(cfg_value(config, "alignment", "stable_mask_radius_px", min(stable_width, stable_height) / 2.0))

    frame_start = args.frame_start
    if frame_start is None:
        frame_start = int(cfg_value(config, "probe", "frame_start", 0))
    frame_count = int(args.frame_count or cfg_value(config, "probe", "frame_count", 120))
    stride = max(1, int(args.stride))
    output = args.output or Path(str(cfg_value(config, "probe", "output_dir", "outputs"))) / "stabilized_clean_lossless.mkv"
    if args.codec == "ffv1" and output.suffix.lower() not in {".mkv", ".avi"}:
        raise ValueError("FFV1 lossless output should use .mkv or .avi; prefer .mkv.")
    status_output = args.status_output or output.with_suffix(".csv")
    ensure_output_dir(output.parent)
    ensure_output_dir(status_output.parent)

    video = get_video_info(crop_video)
    crop_rows = read_crop_meta(crop_meta_csv)
    keypoints = load_keypoint_data(
        zarr_path,
        keypoint_group,
        frame_array=frame_array,
        keypoint_array=keypoint_array,
        valid_array=valid_array,
    )
    selected = selected_crop_rows(
        crop_rows,
        frame_id_column=frame_id_column,
        frame_start=frame_start,
        frame_count=frame_count,
        stride=stride,
    )
    if not selected:
        raise ValueError("No crop rows selected for clean stabilized video.")

    output_fps = max(1.0, float(video.fps) / float(stride)) if np.isfinite(video.fps) and video.fps > 0 else 30.0
    if args.codec == "ffv1":
        writer: VideoSink = FfmpegFfv1VideoSink(
            output,
            fps=output_fps,
            width=stable_width,
            height=stable_height,
            ffmpeg_path=args.ffmpeg_path,
        )
    else:
        writer = OpenCvVideoSink(output, fps=output_fps, width=stable_width, height=stable_height)
    interpolation = _interpolation_flag(args.interpolation)

    capture = cv2.VideoCapture(str(crop_video))
    if not capture.isOpened():
        writer.close()
        raise ValueError(f"Could not open video: {crop_video}")

    status_fields = [
        "output_frame_index",
        "crop_video_frame_index",
        "frame_id_column",
        "frame_id",
        "valid",
        "reason",
        "origin_crop_x",
        "origin_crop_y",
        "forward_angle_deg",
    ]
    rendered = 0
    valid_transforms = 0
    next_expected_index: int | None = None
    blank = np.zeros((stable_height, stable_width, 3), dtype=np.uint8)
    with status_output.open("w", newline="") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=status_fields)
        writer_csv.writeheader()
        try:
            for output_frame_index, (crop_video_index, crop_row) in enumerate(selected):
                frame_id = crop_row_frame_id(crop_video_index, crop_row, frame_id_column)
                keypoint_row = keypoints.frame_to_row.get(frame_id)
                ok, frame, next_expected_index = _read_frame(
                    capture,
                    frame_index=crop_video_index,
                    next_expected_index=next_expected_index,
                )
                valid = False
                reason = "ok"
                origin_xy = np.full(2, np.nan, dtype=np.float64)
                angle = math.nan
                stable = blank
                if not ok:
                    reason = "video_read_failed"
                elif keypoint_row is None:
                    reason = "missing_keypoint_frame"
                elif not bool(keypoints.valid[keypoint_row]):
                    reason = "invalid_keypoints"
                else:
                    kp_crop = keypoints_to_crop_pixels(
                        keypoints.keypoints_img[keypoint_row],
                        crop_row,
                        video_width=video.width,
                        video_height=video.height,
                    )
                    transform = compute_body_transform(
                        kp_crop,
                        stable_width=stable_width,
                        stable_height=stable_height,
                        stable_center_x=stable_center_x,
                        stable_center_y=stable_center_y,
                        origin=origin,
                        target_forward=target_forward,
                        scale=scale,
                        min_forward_length_px=min_forward,
                        min_eye_span_px=min_eye_span,
                    )
                    valid = bool(transform.valid)
                    reason = transform.reason
                    origin_xy = transform.origin_crop_xy
                    angle = transform.forward_angle_deg
                    if valid:
                        stable = cv2.warpAffine(
                            frame,
                            transform.crop_to_stable.astype(np.float32),
                            (stable_width, stable_height),
                            flags=interpolation,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0,
                        )
                        if stable_circular_mask:
                            stable = _apply_circular_mask(
                                stable,
                                center_x=stable_center_x,
                                center_y=stable_center_y,
                                radius_px=stable_mask_radius,
                            )
                        valid_transforms += 1
                    else:
                        stable = blank

                writer.write(stable[:, :, :3])
                writer_csv.writerow(
                    {
                        "output_frame_index": output_frame_index,
                        "crop_video_frame_index": crop_video_index,
                        "frame_id_column": frame_id_column,
                        "frame_id": frame_id,
                        "valid": int(valid),
                        "reason": reason,
                        "origin_crop_x": origin_xy[0],
                        "origin_crop_y": origin_xy[1],
                        "forward_angle_deg": angle,
                    }
                )
                rendered += 1
        finally:
            capture.release()
            writer.close()

    if rendered == 0:
        raise ValueError("No clean stabilized frames were rendered.")
    print(f"wrote: {output}")
    print(f"status_csv: {status_output}")
    print(f"frames_rendered: {rendered}")
    print(f"valid_transforms: {valid_transforms}")
    print(f"fps: {output_fps:g}")
    print(f"origin: {origin}")
    print(f"circular_mask: {int(stable_circular_mask)}")
    print(f"codec: {args.codec}")
    print(f"interpolation: {args.interpolation}")


if __name__ == "__main__":
    main()
