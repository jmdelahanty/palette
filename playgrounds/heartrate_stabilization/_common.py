from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import tomllib
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


KEYPOINT_LABELS = ("swim_bladder", "eye_left", "eye_right", "snout_tip", "tail_tip")
SWIM_BLADDER = 0
EYE_LEFT = 1
EYE_RIGHT = 2


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    frame_count: int
    fps: float


@dataclass(frozen=True)
class KeypointData:
    frame_ids: np.ndarray
    keypoints_img: np.ndarray
    valid: np.ndarray
    frame_to_row: Mapping[int, int]


@dataclass(frozen=True)
class BodyTransform:
    valid: bool
    reason: str
    crop_to_stable: np.ndarray
    stable_to_crop: np.ndarray
    origin_crop_xy: np.ndarray
    forward_angle_deg: float
    scale: float


class SubjectMaskUnavailable(RuntimeError):
    """Raised when a requested subject-mask source cannot be used."""


@dataclass(frozen=True)
class SubjectMaskData:
    parent: str
    run_name: str
    source_path: str
    source_crop_run: str
    component_name: str
    component_index: int
    storage_surface: str
    mask_shape_hw: tuple[int, int]
    source_roi_size_hw: tuple[int, int]
    frame_ids: np.ndarray
    source_crop_row_ids: np.ndarray
    roi_coordinates_full: np.ndarray
    available_channels: np.ndarray | None
    frame_to_rows: Mapping[int, tuple[int, ...]]
    mask_store: Any


@dataclass(frozen=True)
class ProjectedSubjectMask:
    valid: bool
    reason: str
    mask: np.ndarray | None
    mask_row: int
    source_crop_row_id: int
    mask_pixel_count: int


class ThresholdedProbabilityMaskStore:
    """Small dense-reader adapter for raw subject_mask_runs/mask_probs_roi."""

    def __init__(
        self,
        probabilities: Any,
        *,
        labels: Sequence[str],
        thresholds: Sequence[float],
        source_path: str,
    ) -> None:
        raw_shape = tuple(int(value) for value in probabilities.shape)
        if len(raw_shape) == 3:
            shape = (raw_shape[0], 1, raw_shape[1], raw_shape[2])
        elif len(raw_shape) == 4:
            shape = raw_shape
        else:
            raise SubjectMaskUnavailable(f"unsupported_mask_probs_roi_shape:{source_path}:{raw_shape}")
        if len(labels) != int(shape[1]):
            raise SubjectMaskUnavailable(
                f"mask_probability_label_count_mismatch:{source_path}:{len(labels)}!={shape[1]}"
            )
        if len(thresholds) != int(shape[1]):
            raise SubjectMaskUnavailable(
                f"mask_probability_threshold_count_mismatch:{source_path}:{len(thresholds)}!={shape[1]}"
            )
        self.probabilities = probabilities
        self.mask_labels = tuple(str(label) for label in labels)
        self.thresholds = tuple(float(np.clip(value, 0.0, 1.0)) for value in thresholds)
        self.source_path = source_path
        self.shape = (int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3]))
        self.storage_surface = "mask_probs_roi"

    def component_index(self, component_name: str) -> int:
        try:
            return self.mask_labels.index(str(component_name))
        except ValueError as exc:
            raise SubjectMaskUnavailable(
                f"missing_mask_component:{component_name}:available={','.join(self.mask_labels)}"
            ) from exc

    def _indices(self, values: Sequence[int] | np.ndarray | slice | int | None, size: int) -> np.ndarray:
        if values is None:
            return np.arange(int(size), dtype=np.int64)
        if isinstance(values, slice):
            return np.arange(int(size), dtype=np.int64)[values]
        if isinstance(values, (int, np.integer)):
            out = np.asarray([int(values)], dtype=np.int64)
        else:
            out = np.asarray(list(values), dtype=np.int64).reshape(-1)
        if np.any(out < 0) or np.any(out >= int(size)):
            raise SubjectMaskUnavailable(f"mask_probability_index_out_of_bounds:{self.source_path}")
        return out

    def _channel_indices(
        self,
        channels: Sequence[int | str] | int | str | slice | None,
    ) -> np.ndarray:
        if channels is None or isinstance(channels, slice):
            return self._indices(channels, int(self.shape[1]))
        if isinstance(channels, str):
            return np.asarray([self.component_index(channels)], dtype=np.int64)
        if isinstance(channels, (int, np.integer)):
            return self._indices(int(channels), int(self.shape[1]))
        resolved: list[int] = []
        for value in channels:
            if isinstance(value, str):
                resolved.append(self.component_index(value))
            else:
                resolved.append(int(value))
        return self._indices(resolved, int(self.shape[1]))

    def _decode_probabilities(self, raw: np.ndarray) -> np.ndarray:
        probs = np.asarray(raw, dtype=np.float32)
        if np.issubdtype(np.asarray(raw).dtype, np.integer):
            max_value = float(np.iinfo(np.asarray(raw).dtype).max)
            if max_value > 0:
                probs = probs / max_value
        return probs

    def read_dense(
        self,
        rows: Sequence[int] | np.ndarray | slice | int | None = None,
        channels: Sequence[int | str] | int | str | slice | None = None,
    ) -> np.ndarray:
        row_indices = self._indices(rows, int(self.shape[0]))
        channel_indices = self._channel_indices(channels)
        output = np.zeros((int(row_indices.size), int(channel_indices.size), *self.shape[2:]), dtype=np.uint8)
        source_ndim = len(tuple(int(value) for value in self.probabilities.shape))
        for out_row, row_idx in enumerate(row_indices):
            for out_channel, channel_idx in enumerate(channel_indices):
                if source_ndim == 3:
                    raw = np.asarray(self.probabilities[int(row_idx)], dtype=np.float32)
                else:
                    raw = np.asarray(self.probabilities[int(row_idx), int(channel_idx)])
                probs = self._decode_probabilities(raw)
                output[out_row, out_channel] = (probs >= self.thresholds[int(channel_idx)]).astype(np.uint8)
        return output


def load_config(path: Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def cfg_path(config: Mapping[str, Any], section: str, key: str) -> Path:
    value = config.get(section, {}).get(key)
    if not value:
        raise ValueError(f"Missing config value [{section}].{key}")
    return Path(str(value)).expanduser()


def cfg_value(config: Mapping[str, Any], section: str, key: str, default: Any = None) -> Any:
    return config.get(section, {}).get(key, default)


def read_crop_meta(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def row_float(row: Mapping[str, str], key: str, default: float = math.nan) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def row_int(row: Mapping[str, str], key: str, default: int = -1) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return int(default)


def crop_row_frame_id(row_index: int, row: Mapping[str, str], frame_id_column: str) -> int:
    normalized = str(frame_id_column).strip().lower()
    if normalized in {"crop_video_frame_index", "row_index", "csv_row_index"}:
        return int(row_index)
    if normalized in {"recording_frame_id_zero_based", "recording_frame_index"}:
        return row_int(row, "recording_frame_id") - 1
    return row_int(row, frame_id_column)


def get_video_info(path: Path) -> VideoInfo:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    return VideoInfo(width=width, height=height, frame_count=frame_count, fps=fps)


def zarr_json_metadata(zarr_path: Path, child_path: str) -> Mapping[str, Any]:
    metadata_path = Path(zarr_path).joinpath(*[part for part in child_path.split("/") if part], "zarr.json")
    with metadata_path.open("r") as handle:
        return json.load(handle)


def _resolve_latest_run(parent: Any, label: str) -> str:
    latest = parent.attrs.get("latest") if hasattr(parent, "attrs") else None
    if latest:
        return str(latest)
    keys = sorted(str(key) for key in parent.keys())
    if not keys:
        raise SubjectMaskUnavailable(f"no_runs:{label}")
    return keys[-1]


def _read_array_data(array: Any) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return np.asarray(array)
    return np.asarray(array[:])


def _frame_to_rows(frame_ids: np.ndarray) -> Mapping[int, tuple[int, ...]]:
    pending: dict[int, list[int]] = {}
    for row_index, frame_id in enumerate(np.asarray(frame_ids, dtype=np.int64).reshape(-1).tolist()):
        pending.setdefault(int(frame_id), []).append(int(row_index))
    return {frame_id: tuple(rows) for frame_id, rows in pending.items()}


def _mask_labels_from_attrs(attrs: Mapping[str, Any], channel_count: int) -> tuple[str, ...]:
    for key in ("mask_labels", "component_names", "labels"):
        raw = attrs.get(key)
        if isinstance(raw, (list, tuple)) and raw:
            labels = tuple(str(value) for value in raw)
            if len(labels) == int(channel_count):
                return labels
    return tuple(f"component_{idx}" for idx in range(int(channel_count)))


def _coerce_probability_threshold(value: object, *, default: float = 0.5) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        threshold = float(default)
    return float(np.clip(threshold, 0.0, 1.0))


def _probability_thresholds_for_labels(attrs: Mapping[str, Any], labels: Sequence[str]) -> tuple[float, ...]:
    default_threshold = _coerce_probability_threshold(attrs.get("mask_probability_threshold"), default=0.5)
    thresholds = [default_threshold for _label in labels]
    raw = (
        attrs.get("thresholds_by_label")
        or attrs.get("threshold_by_component")
        or attrs.get("threshold_by_label")
    )
    if isinstance(raw, Mapping):
        for idx, label in enumerate(labels):
            value = raw.get(str(label))
            if value is not None:
                thresholds[idx] = _coerce_probability_threshold(value, default=default_threshold)
    elif isinstance(raw, (list, tuple)) and len(raw) == len(labels):
        thresholds = [_coerce_probability_threshold(value, default=default_threshold) for value in raw]
    return tuple(float(value) for value in thresholds)


def load_subject_mask_data(
    zarr_path: Path,
    *,
    parent: str = "refined_subject_masks_runs",
    run_name: str = "latest",
    component_name: str = "subject_body",
    allow_stale_rle: bool = False,
) -> SubjectMaskData:
    import zarr

    from fisheye.shared.mask_store import MaskStoreError, open_mask_store
    from fisheye.shared.row_lineage import resolve_source_crop_row_ids

    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    parent_name = str(parent).strip("/")
    parent_group = root.get(parent_name)
    if parent_group is None:
        raise SubjectMaskUnavailable(f"missing_mask_parent:{parent_name}")

    requested_run = str(run_name or "latest")
    resolved_run = _resolve_latest_run(parent_group, parent_name) if requested_run == "latest" else requested_run
    run_group = parent_group.get(resolved_run)
    if run_group is None:
        raise SubjectMaskUnavailable(f"missing_mask_run:{parent_name}/{resolved_run}")

    source_path = f"{parent_name}/{resolved_run}"
    try:
        mask_store = open_mask_store(
            run_group,
            source_path=source_path,
            prefer="auto",
            allow_stale_rle=bool(allow_stale_rle),
        )
    except MaskStoreError as exc:
        probabilities = run_group.get("mask_probs_roi")
        if probabilities is None:
            raise SubjectMaskUnavailable(f"unreadable_mask_store:{source_path}:{exc}") from exc
        probability_shape = tuple(int(value) for value in probabilities.shape)
        if len(probability_shape) == 3:
            channel_count = 1
        elif len(probability_shape) == 4:
            channel_count = int(probability_shape[1])
        else:
            raise SubjectMaskUnavailable(f"unsupported_mask_probs_roi_shape:{source_path}:{probability_shape}") from exc
        probability_labels = _mask_labels_from_attrs(run_group.attrs, channel_count)
        mask_store = ThresholdedProbabilityMaskStore(
            probabilities,
            labels=probability_labels,
            thresholds=_probability_thresholds_for_labels(run_group.attrs, probability_labels),
            source_path=source_path,
        )

    labels = tuple(str(label) for label in mask_store.mask_labels)
    requested_component = str(component_name)
    if requested_component not in labels:
        available = ",".join(labels) if labels else "<none>"
        raise SubjectMaskUnavailable(f"missing_mask_component:{requested_component}:available={available}")
    component_index = int(mask_store.component_index(requested_component))

    total_rows = int(mask_store.shape[0])
    frame_array = run_group.get("frame_indices")
    if frame_array is None:
        raise SubjectMaskUnavailable(f"missing_frame_indices:{source_path}")
    frame_ids = np.asarray(frame_array[:], dtype=np.int64).reshape(-1)
    if int(frame_ids.shape[0]) != total_rows:
        raise SubjectMaskUnavailable(
            f"mask_frame_indices_length_mismatch:{source_path}:{frame_ids.shape[0]}!={total_rows}"
        )

    crop_run = str(run_group.attrs.get("source_crop_run") or "")
    crop_parent = root.get("crop_runs")
    if not crop_run and crop_parent is not None:
        crop_run = _resolve_latest_run(crop_parent, "crop_runs")
    if not crop_run:
        raise SubjectMaskUnavailable(f"missing_source_crop_run:{source_path}")
    crop_group = root.get(f"crop_runs/{crop_run}")
    if crop_group is None:
        raise SubjectMaskUnavailable(f"missing_source_crop_run_group:crop_runs/{crop_run}")
    if "roi_coordinates_full" not in crop_group:
        raise SubjectMaskUnavailable(f"missing_roi_coordinates_full:crop_runs/{crop_run}")
    roi_size_attr = crop_group.attrs.get("roi_size")
    if isinstance(roi_size_attr, (list, tuple)) and len(roi_size_attr) == 2:
        source_roi_size_hw = (int(roi_size_attr[0]), int(roi_size_attr[1]))
    elif "roi_images" in crop_group and len(crop_group["roi_images"].shape) >= 3:
        roi_images_shape = crop_group["roi_images"].shape
        source_roi_size_hw = (int(roi_images_shape[1]), int(roi_images_shape[2]))
    else:
        raise SubjectMaskUnavailable(f"missing_source_crop_roi_size:crop_runs/{crop_run}")

    source_crop_row_ids_raw = resolve_source_crop_row_ids(
        run_group,
        crop_group,
        total_rois=total_rows,
        frame_indices=frame_array,
    )
    if source_crop_row_ids_raw is None:
        raise SubjectMaskUnavailable(f"missing_source_crop_row_ids:{source_path}")
    source_crop_row_ids = np.asarray(_read_array_data(source_crop_row_ids_raw), dtype=np.int64).reshape(-1)
    if int(source_crop_row_ids.shape[0]) != total_rows:
        raise SubjectMaskUnavailable(
            f"source_crop_row_ids_length_mismatch:{source_path}:{source_crop_row_ids.shape[0]}!={total_rows}"
        )

    all_roi_coordinates = np.asarray(crop_group["roi_coordinates_full"][:], dtype=np.float64)
    if all_roi_coordinates.ndim != 2 or all_roi_coordinates.shape[1] < 2:
        raise SubjectMaskUnavailable(f"invalid_roi_coordinates_full_shape:crop_runs/{crop_run}")
    if source_crop_row_ids.size:
        min_row = int(source_crop_row_ids.min())
        max_row = int(source_crop_row_ids.max())
        if min_row < 0 or max_row >= int(all_roi_coordinates.shape[0]):
            raise SubjectMaskUnavailable(f"source_crop_row_ids_out_of_bounds:{source_path}")
    roi_coordinates = all_roi_coordinates[source_crop_row_ids, :2]

    available_channels = None
    available_array = run_group.get("available_channels")
    if available_array is not None:
        available_channels = np.asarray(available_array[:], dtype=bool)

    return SubjectMaskData(
        parent=parent_name,
        run_name=resolved_run,
        source_path=source_path,
        source_crop_run=crop_run,
        component_name=requested_component,
        component_index=component_index,
        storage_surface=str(mask_store.storage_surface),
        mask_shape_hw=(int(mask_store.shape[2]), int(mask_store.shape[3])),
        source_roi_size_hw=source_roi_size_hw,
        frame_ids=frame_ids,
        source_crop_row_ids=source_crop_row_ids,
        roi_coordinates_full=roi_coordinates,
        available_channels=available_channels,
        frame_to_rows=_frame_to_rows(frame_ids),
        mask_store=mask_store,
    )


def load_keypoint_data(
    zarr_path: Path,
    keypoint_group_path: str,
    *,
    frame_array: str = "frame_indices",
    keypoint_array: str = "keypoints_img",
    valid_array: str = "usable_keypoints",
) -> KeypointData:
    import zarr

    root = zarr.open_group(str(zarr_path), mode="r", use_consolidated=False)
    group = root["/".join(part for part in keypoint_group_path.strip("/").split("/") if part)]
    frame_ids = np.asarray(group[frame_array][:], dtype=np.int64).reshape(-1)
    keypoints = np.asarray(group[keypoint_array][:], dtype=np.float64)
    if keypoints.ndim != 3 or keypoints.shape[2] != 2:
        raise ValueError(f"{keypoint_group_path}/{keypoint_array} must have shape (N, K, 2), got {keypoints.shape}")
    n = min(frame_ids.shape[0], keypoints.shape[0])
    frame_ids = frame_ids[:n]
    keypoints = keypoints[:n]

    valid = np.isfinite(keypoints[:, [SWIM_BLADDER, EYE_LEFT, EYE_RIGHT], :]).all(axis=(1, 2))
    if valid_array and valid_array in group:
        raw_valid = np.asarray(group[valid_array][:])
        raw_valid = raw_valid[:n]
        if raw_valid.ndim == 1:
            valid &= raw_valid.astype(bool)
        else:
            valid &= raw_valid.reshape(raw_valid.shape[0], -1).all(axis=1).astype(bool)

    frame_to_row: dict[int, int] = {}
    for row_index, frame_id in enumerate(frame_ids.tolist()):
        frame_to_row.setdefault(int(frame_id), int(row_index))
    return KeypointData(frame_ids=frame_ids, keypoints_img=keypoints, valid=valid, frame_to_row=frame_to_row)


def selected_crop_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    frame_id_column: str,
    frame_start: int | None,
    frame_count: int,
    stride: int,
) -> list[tuple[int, Mapping[str, str]]]:
    if frame_start is None:
        start_index = 0
    else:
        start_index = next(
            (
                idx
                for idx, row in enumerate(rows)
                if crop_row_frame_id(idx, row, frame_id_column) >= int(frame_start)
            ),
            len(rows),
        )
    stop_index = min(len(rows), start_index + max(0, int(frame_count)))
    step = max(1, int(stride))
    return [(idx, rows[idx]) for idx in range(start_index, stop_index, step)]


def keypoints_to_crop_pixels(
    keypoints_img: np.ndarray,
    crop_row: Mapping[str, str],
    *,
    video_width: int,
    video_height: int,
) -> np.ndarray:
    crop_x = row_float(crop_row, "crop_x")
    crop_y = row_float(crop_row, "crop_y")
    crop_w = row_float(crop_row, "crop_w", float(video_width))
    crop_h = row_float(crop_row, "crop_h", float(video_height))
    scale_x = float(video_width) / crop_w if np.isfinite(crop_w) and crop_w > 0 else 1.0
    scale_y = float(video_height) / crop_h if np.isfinite(crop_h) and crop_h > 0 else 1.0
    out = np.asarray(keypoints_img, dtype=np.float64).copy()
    out[:, 0] = (out[:, 0] - crop_x) * scale_x
    out[:, 1] = (out[:, 1] - crop_y) * scale_y
    return out


def _target_forward_vector(name: str) -> np.ndarray:
    normalized = str(name).strip().lower()
    if normalized == "up":
        return np.asarray([0.0, -1.0], dtype=np.float64)
    if normalized == "right":
        return np.asarray([1.0, 0.0], dtype=np.float64)
    if normalized == "down":
        return np.asarray([0.0, 1.0], dtype=np.float64)
    if normalized == "left":
        return np.asarray([-1.0, 0.0], dtype=np.float64)
    raise ValueError(f"Unsupported target_forward={name!r}")


def _invert_affine(matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.eye(3, dtype=np.float64)
    homogeneous[:2, :] = np.asarray(matrix, dtype=np.float64)
    inverse = np.linalg.inv(homogeneous)
    return inverse[:2, :]


def transform_points(matrix: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64)
    flat = points.reshape(-1, 2)
    ones = np.ones((flat.shape[0], 1), dtype=np.float64)
    transformed = np.column_stack([flat, ones]) @ np.asarray(matrix, dtype=np.float64).T
    return transformed.reshape(points.shape)


def compute_body_transform(
    keypoints_crop_xy: np.ndarray,
    *,
    stable_width: int,
    stable_height: int,
    stable_center_x: float,
    stable_center_y: float,
    origin: str = "eye_midpoint",
    target_forward: str = "up",
    scale: float = 1.0,
    min_forward_length_px: float = 8.0,
    min_eye_span_px: float = 4.0,
) -> BodyTransform:
    empty = np.asarray([[math.nan, math.nan, math.nan], [math.nan, math.nan, math.nan]], dtype=np.float64)
    keypoints = np.asarray(keypoints_crop_xy, dtype=np.float64)
    required = keypoints[[SWIM_BLADDER, EYE_LEFT, EYE_RIGHT], :]
    if not np.isfinite(required).all():
        return BodyTransform(False, "nonfinite_required_keypoints", empty, empty, np.full(2, np.nan), math.nan, scale)

    swim = keypoints[SWIM_BLADDER]
    eye_mid = np.mean(keypoints[[EYE_LEFT, EYE_RIGHT]], axis=0)
    eye_span = float(np.linalg.norm(keypoints[EYE_LEFT] - keypoints[EYE_RIGHT]))
    if not np.isfinite(eye_span) or eye_span < float(min_eye_span_px):
        return BodyTransform(False, "eye_span_too_small", empty, empty, eye_mid, math.nan, scale)

    forward = eye_mid - swim
    forward_length = float(np.linalg.norm(forward))
    if not np.isfinite(forward_length) or forward_length < float(min_forward_length_px):
        return BodyTransform(False, "forward_axis_too_short", empty, empty, eye_mid, math.nan, scale)

    normalized_origin = str(origin).strip().lower()
    if normalized_origin == "eye_midpoint":
        origin_xy = eye_mid
    elif normalized_origin == "swim_bladder":
        origin_xy = swim
    elif normalized_origin in {"eye_swim_midpoint", "swim_eye_midpoint", "body_midpoint"}:
        origin_xy = 0.5 * (eye_mid + swim)
    elif normalized_origin == "head_triplet_mean":
        origin_xy = np.mean(required, axis=0)
    else:
        raise ValueError(f"Unsupported origin={origin!r}")

    source_unit = forward / forward_length
    target_unit = _target_forward_vector(target_forward)
    cosine = float(np.dot(source_unit, target_unit))
    sine = float(source_unit[0] * target_unit[1] - source_unit[1] * target_unit[0])
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    effective_scale = float(scale)
    if not np.isfinite(effective_scale) or effective_scale <= 0:
        effective_scale = 1.0
    linear = effective_scale * rotation
    center = np.asarray([float(stable_center_x), float(stable_center_y)], dtype=np.float64)
    translation = center - linear @ origin_xy
    crop_to_stable = np.column_stack([linear, translation])
    stable_to_crop = _invert_affine(crop_to_stable)
    angle = math.degrees(math.atan2(-source_unit[1], source_unit[0]))
    return BodyTransform(
        True,
        "ok",
        crop_to_stable,
        stable_to_crop,
        origin_xy.astype(np.float64),
        float(angle),
        effective_scale,
    )


def roi_rect_corners(rect: Sequence[float]) -> np.ndarray:
    if len(rect) != 4:
        raise ValueError("ROI rectangle must be x,y,width,height")
    x, y, w, h = [float(value) for value in rect]
    return np.asarray(
        [
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h],
        ],
        dtype=np.float64,
    )


def polygon_mask(shape_hw: Sequence[int], polygon_xy: np.ndarray) -> np.ndarray:
    import cv2

    polygon = np.asarray(polygon_xy, dtype=np.float64)
    finite = np.isfinite(polygon).all(axis=1)
    shape = tuple(int(value) for value in shape_hw[:2])
    mask = np.zeros(shape, dtype=np.uint8)
    if polygon.shape[0] < 3 or not np.all(finite):
        return mask.astype(bool)
    rounded = np.round(polygon).astype(np.int32).reshape(1, -1, 2)
    cv2.fillPoly(mask, rounded, 255)
    return mask > 0


def mask_mean_intensity(frame: np.ndarray, sample_mask: np.ndarray) -> tuple[float, int]:
    import cv2

    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    mask = np.asarray(sample_mask, dtype=bool)
    if mask.shape != gray.shape[:2]:
        raise ValueError(f"sample mask shape {mask.shape} does not match frame shape {gray.shape[:2]}")
    values = gray[mask]
    if values.size == 0:
        return math.nan, 0
    return float(np.mean(values)), int(values.size)


def polygon_mean_intensity(frame: np.ndarray, polygon_xy: np.ndarray) -> tuple[float, int]:
    return mask_mean_intensity(frame, polygon_mask(frame.shape[:2], polygon_xy))


def project_subject_mask_to_crop_frame(
    mask_data: SubjectMaskData,
    *,
    frame_id: int,
    crop_row: Mapping[str, str],
    video_width: int,
    video_height: int,
) -> ProjectedSubjectMask:
    import cv2

    matching_rows = mask_data.frame_to_rows.get(int(frame_id))
    if not matching_rows:
        return ProjectedSubjectMask(False, "missing_mask_frame", None, -1, -1, 0)
    mask_row = int(matching_rows[0])

    available = mask_data.available_channels
    if available is not None:
        channel = int(mask_data.component_index)
        if available.ndim == 1:
            if channel >= int(available.shape[0]) or not bool(available[channel]):
                return ProjectedSubjectMask(False, "mask_component_unavailable", None, mask_row, -1, 0)
        elif available.ndim >= 2:
            if (
                mask_row >= int(available.shape[0])
                or channel >= int(available.shape[1])
                or not bool(available[mask_row, channel])
            ):
                return ProjectedSubjectMask(False, "mask_component_unavailable", None, mask_row, -1, 0)

    try:
        mask_roi = mask_data.mask_store.read_dense(
            rows=mask_row,
            channels=mask_data.component_name,
        )[0, 0]
    except Exception as exc:
        return ProjectedSubjectMask(False, f"mask_read_failed:{exc}", None, mask_row, -1, 0)

    mask_roi = (np.asarray(mask_roi, dtype=np.uint8) > 0).astype(np.uint8)
    if mask_roi.size == 0 or int(np.count_nonzero(mask_roi)) == 0:
        return ProjectedSubjectMask(False, "empty_mask_roi", None, mask_row, -1, 0)

    crop_x = row_float(crop_row, "crop_x")
    crop_y = row_float(crop_row, "crop_y")
    crop_w = row_float(crop_row, "crop_w", float(video_width))
    crop_h = row_float(crop_row, "crop_h", float(video_height))
    if not np.isfinite([crop_x, crop_y, crop_w, crop_h]).all() or crop_w <= 0 or crop_h <= 0:
        return ProjectedSubjectMask(False, "invalid_crop_geometry", None, mask_row, -1, 0)

    source_crop_row_id = int(mask_data.source_crop_row_ids[mask_row])
    roi_x, roi_y = [float(value) for value in mask_data.roi_coordinates_full[mask_row, :2]]
    source_roi_h, source_roi_w = mask_data.source_roi_size_hw
    mask_h, mask_w = mask_roi.shape[:2]
    if source_roi_h <= 0 or source_roi_w <= 0 or mask_h <= 0 or mask_w <= 0:
        return ProjectedSubjectMask(False, "invalid_mask_geometry", None, mask_row, source_crop_row_id, 0)

    full_per_mask_x = float(source_roi_w) / float(mask_w)
    full_per_mask_y = float(source_roi_h) / float(mask_h)
    crop_video_scale_x = float(video_width) / float(crop_w)
    crop_video_scale_y = float(video_height) / float(crop_h)
    affine = np.asarray(
        [
            [full_per_mask_x * crop_video_scale_x, 0.0, (roi_x - crop_x) * crop_video_scale_x],
            [0.0, full_per_mask_y * crop_video_scale_y, (roi_y - crop_y) * crop_video_scale_y],
        ],
        dtype=np.float32,
    )
    projected = cv2.warpAffine(
        mask_roi * 255,
        affine,
        (int(video_width), int(video_height)),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    projected_mask = projected > 0
    pixel_count = int(np.count_nonzero(projected_mask))
    if pixel_count == 0:
        return ProjectedSubjectMask(False, "projected_mask_outside_crop", projected_mask, mask_row, source_crop_row_id, 0)
    reason = "ok" if len(matching_rows) == 1 else "ok_multiple_mask_rows_first_used"
    return ProjectedSubjectMask(True, reason, projected_mask, mask_row, source_crop_row_id, pixel_count)


def parse_roi_rect(raw: str | Sequence[float]) -> tuple[float, float, float, float]:
    if isinstance(raw, str):
        values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    else:
        values = [float(part) for part in raw]
    if len(values) != 4:
        raise ValueError("ROI must contain four numbers: x,y,width,height")
    return tuple(values)  # type: ignore[return-value]


def load_roi_rect_json(path: Path) -> tuple[float, float, float, float]:
    with Path(path).open("r") as handle:
        payload = json.load(handle)
    for key in ("roi_rect_stable_xywh", "rect", "roi"):
        if key in payload:
            return parse_roi_rect(payload[key])
    raise ValueError(f"ROI JSON does not contain roi_rect_stable_xywh, rect, or roi: {path}")


def resolve_roi_rect(
    config: Mapping[str, Any],
    *,
    roi: str | Sequence[float] | None = None,
    roi_json: Path | None = None,
) -> tuple[float, float, float, float]:
    if roi is not None:
        return parse_roi_rect(roi)
    if roi_json is not None:
        return load_roi_rect_json(roi_json)
    return parse_roi_rect(cfg_value(config, "roi", "rect"))


def ensure_output_dir(path: Path) -> Path:
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)
