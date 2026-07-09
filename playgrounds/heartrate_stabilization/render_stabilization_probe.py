from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _common import (
    EYE_LEFT,
    EYE_RIGHT,
    KEYPOINT_LABELS,
    SWIM_BLADDER,
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
    transform_points,
)


def _draw_keypoints(image: np.ndarray, keypoints_xy: np.ndarray, *, color: tuple[int, int, int]) -> np.ndarray:
    import cv2

    out = image.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    for idx, point in enumerate(np.asarray(keypoints_xy, dtype=np.float64)):
        if not np.isfinite(point).all():
            continue
        x, y = int(round(point[0])), int(round(point[1]))
        cv2.circle(out, (x, y), 3, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(out, str(idx), (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    if np.isfinite(keypoints_xy[[SWIM_BLADDER, EYE_LEFT, EYE_RIGHT], :]).all():
        swim = tuple(np.round(keypoints_xy[SWIM_BLADDER]).astype(int).tolist())
        eye_mid = tuple(np.round(np.mean(keypoints_xy[[EYE_LEFT, EYE_RIGHT]], axis=0)).astype(int).tolist())
        cv2.line(out, swim, eye_mid, (0, 255, 255), 1, lineType=cv2.LINE_AA)
    return out


def _label_panel(image: np.ndarray, text: str) -> np.ndarray:
    import cv2

    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _make_contact_sheet(images: list[np.ndarray], *, columns: int = 2) -> np.ndarray:
    if not images:
        raise ValueError("No images for contact sheet.")
    h = max(image.shape[0] for image in images)
    w = max(image.shape[1] for image in images)
    c = 3
    rows = int(np.ceil(len(images) / float(columns)))
    sheet = np.zeros((rows * h, columns * w, c), dtype=np.uint8)
    for idx, image in enumerate(images):
        row = idx // columns
        col = idx % columns
        panel = image
        if panel.ndim == 2:
            import cv2

            panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
        sheet[row * h : row * h + panel.shape[0], col * w : col * w + panel.shape[1], :] = panel[:, :, :3]
    return sheet


def main() -> None:
    import cv2

    parser = argparse.ArgumentParser(description="Render fish-body stabilization probe frames.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.example.toml"))
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-count", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-panels", type=int, default=24)
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
    frame_start = args.frame_start
    if frame_start is None:
        frame_start = int(cfg_value(config, "probe", "frame_start", 0))
    frame_count = int(args.frame_count or cfg_value(config, "probe", "frame_count", 120))
    stride = int(args.stride or cfg_value(config, "probe", "stride", 10))
    output_dir = ensure_output_dir(args.output_dir or Path(str(cfg_value(config, "probe", "output_dir", "outputs"))))

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
    )[: max(1, int(args.max_panels))]

    capture = cv2.VideoCapture(str(crop_video))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {crop_video}")

    panels: list[np.ndarray] = []
    try:
        for crop_video_index, crop_row in selected:
            frame_id = crop_row_frame_id(crop_video_index, crop_row, frame_id_column)
            keypoint_row = keypoints.frame_to_row.get(frame_id)
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(crop_video_index))
            ok, frame = capture.read()
            if not ok:
                continue
            if keypoint_row is None or not bool(keypoints.valid[keypoint_row]):
                crop_panel = _label_panel(frame, f"crop frame={frame_id} no valid keypoints")
                stable_panel = np.zeros((stable_height, stable_width, 3), dtype=np.uint8)
                stable_panel = _label_panel(stable_panel, "stable unavailable")
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
                crop_panel = _draw_keypoints(frame, kp_crop, color=(0, 255, 0))
                crop_panel = _label_panel(crop_panel, f"crop frame={frame_id}")
                if transform.valid:
                    stable = cv2.warpAffine(
                        frame,
                        transform.crop_to_stable.astype(np.float32),
                        (stable_width, stable_height),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0,
                    )
                    kp_stable = transform_points(transform.crop_to_stable, kp_crop)
                    stable_panel = _draw_keypoints(stable, kp_stable, color=(0, 255, 0))
                    stable_panel = _label_panel(stable_panel, f"stable angle={transform.forward_angle_deg:.1f}")
                else:
                    stable_panel = np.zeros((stable_height, stable_width, 3), dtype=np.uint8)
                    stable_panel = _label_panel(stable_panel, f"stable invalid: {transform.reason}")

            combined = np.hstack([crop_panel, stable_panel])
            out_path = output_dir / f"stabilization_probe_{crop_video_index:06d}_{frame_id}.png"
            cv2.imwrite(str(out_path), combined)
            panels.append(combined)
    finally:
        capture.release()

    if panels:
        sheet = _make_contact_sheet(panels, columns=2)
        sheet_path = output_dir / "stabilization_probe_contact_sheet.png"
        cv2.imwrite(str(sheet_path), sheet)
        print(f"wrote {len(panels)} frame probes")
        print(f"contact_sheet: {sheet_path}")
        print(f"keypoints: {', '.join(f'{i}:{name}' for i, name in enumerate(KEYPOINT_LABELS))}")
    else:
        print("No probe frames were rendered.")


if __name__ == "__main__":
    main()
