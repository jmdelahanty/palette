"""YOLO-based eye segmentation for Palette ROI crops."""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime, timezone
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import zarr
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn
from skimage import measure

from ..shared.provenance_attrs import build_source_keypoints_attrs, resolve_source_keypoints_run
from ..shared.row_alignment import assert_row_alignment
from ..shared.zarr.schema import get_run_group
from ..utils.system import get_environment_info, get_git_info


@dataclass
class EyeCandidate:
    mask: np.ndarray
    prob_mask: Optional[np.ndarray]
    centroid_xy: Tuple[float, float]
    ellipse_row: np.ndarray
    contour_xy: Optional[np.ndarray]
    area: float


def _prepare_run_group(
    root: zarr.Group,
    run_name: Optional[str],
    console: Console,
) -> Tuple[zarr.Group, str]:
    parent = root.require_group("eye_masks_runs")
    if run_name:
        if run_name in parent:
            raise ValueError(f"eye_masks_runs/{run_name} already exists")
        run_group = parent.create_group(run_name)
        parent.attrs["latest"] = run_name
        console.print(f"Created run group: [cyan]eye_masks_runs/{run_name}[/cyan]")
        return run_group, run_name
    return get_run_group(root, "eye_masks", console=console, create_new=True)


def _as_optional_text(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore")
    return str(value)


def _resolve_keypoint_lineage(
    root: zarr.Group,
    crop_group: zarr.Group,
    keypoints_run: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    refined_parent = root.get("refined_keypoints_runs")
    raw_parent = root.get("keypoints_runs")

    requested_run = _as_optional_text(keypoints_run)
    if requested_run:
        if refined_parent is not None and requested_run in refined_parent:
            return requested_run, "refined_keypoints_runs"
        if raw_parent is not None and requested_run in raw_parent:
            return requested_run, "keypoints_runs"
        return requested_run, _as_optional_text(crop_group.attrs.get("source_keypoint_group"))

    source_group_name = _as_optional_text(crop_group.attrs.get("source_keypoint_group"))
    source_run_name = _as_optional_text(resolve_source_keypoints_run(crop_group.attrs))

    if source_group_name and source_run_name:
        source_parent = root.get(source_group_name)
        if source_parent is not None and source_run_name in source_parent:
            return source_run_name, source_group_name

    if source_run_name:
        if refined_parent is not None and source_run_name in refined_parent:
            return source_run_name, "refined_keypoints_runs"
        if raw_parent is not None and source_run_name in raw_parent:
            return source_run_name, "keypoints_runs"
        return source_run_name, source_group_name

    refined_latest = _as_optional_text(refined_parent.attrs.get("latest")) if refined_parent is not None else None
    raw_latest = _as_optional_text(raw_parent.attrs.get("latest")) if raw_parent is not None else None

    if refined_parent is not None and refined_latest in refined_parent:
        return refined_latest, "refined_keypoints_runs"
    if raw_parent is not None and raw_latest in raw_parent:
        return raw_latest, "keypoints_runs"
    return None, source_group_name


def _validate_input_row_alignment(
    *,
    crop_group: zarr.Group,
    crop_run_name: str,
    total_rois: int,
) -> None:
    assert_row_alignment(
        total_rois,
        (
            (f"crop_runs/{crop_run_name}/roi_images", crop_group["roi_images"]),
            (f"crop_runs/{crop_run_name}/frame_indices", crop_group.get("frame_indices")),
            (f"crop_runs/{crop_run_name}/detection_indices", crop_group.get("detection_indices")),
            (f"crop_runs/{crop_run_name}/detection_source", crop_group.get("detection_source")),
        ),
        stage="eye_segmentation_yolo input",
    )


_MASK_PROB_CACHE: Deque[torch.Tensor] = deque()
_PROCESS_MASK_PATCHED = False
_ORIGINAL_PROCESS_MASK_NATIVE = None
_PROTO_UPSAMPLE_FACTOR = 1
_DEFAULT_LEGACY_MASKS = False


def _shape_to_hw(shape) -> Tuple[int, int]:
    if isinstance(shape, torch.Size):
        shape = tuple(shape)
    if isinstance(shape, (tuple, list)):
        if len(shape) >= 2:
            return int(shape[0]), int(shape[1])
        if len(shape) == 1:
            val = int(shape[0])
            return val, val
        return 0, 0
    try:
        seq = tuple(shape)
        if len(seq) >= 2:
            return int(seq[0]), int(seq[1])
        if len(seq) == 1:
            val = int(seq[0])
            return val, val
    except TypeError:
        pass
    val = int(shape)
    return val, val


def _prepare_proto_tensor(
    protos,
    upsample_factor: int,
) -> Optional[torch.Tensor]:
    if protos is None:
        return None
    if isinstance(protos, (list, tuple)):
        if not protos:
            return None
        return _prepare_proto_tensor(protos[0], upsample_factor)
    if torch.is_tensor(protos):
        tensor = protos
    else:
        tensor = torch.as_tensor(protos)
    if upsample_factor > 1:
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            scale_factor=upsample_factor,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return tensor.contiguous()


def _compute_native_soft_masks(
    proto_tensor: torch.Tensor,
    coeffs,
    boxes,
    shape,
    upsample: bool = True,
) -> Optional[torch.Tensor]:
    if proto_tensor is None or coeffs is None or shape is None:
        return None
    try:
        protos_t = proto_tensor
        if protos_t.ndim != 3:
            return None
        c, mh, mw = protos_t.shape
        coeffs_t = coeffs if torch.is_tensor(coeffs) else torch.as_tensor(coeffs)
        if coeffs_t.numel() == 0:
            return None
        coeffs_t = coeffs_t.to(protos_t.device).to(protos_t.dtype)
        if coeffs_t.ndim == 1:
            coeffs_t = coeffs_t.unsqueeze(0)
        masks = coeffs_t @ protos_t.reshape(c, -1)
        masks = masks.sigmoid().view(-1, mh, mw)
        boxes_t = boxes if torch.is_tensor(boxes) else torch.as_tensor(boxes)
        if boxes_t.numel() and boxes_t.shape[0] == masks.shape[0]:
            try:
                from ultralytics.utils import ops as _ops

                boxes_proc = boxes_t.to(protos_t.device).to(protos_t.dtype)
                shape_src = _shape_to_hw(shape)
                scaled = _ops.scale_boxes(shape_src, boxes_proc, (mh, mw))
                masks = _ops.crop_mask(masks, scaled)
            except Exception:
                pass
        if upsample:
            ih, iw = _shape_to_hw(shape)
            masks = F.interpolate(masks.unsqueeze(0), size=(int(ih), int(iw)), mode="bilinear", align_corners=False).squeeze(0)
        return masks.clamp_(0.0, 1.0)
    except Exception:
        return None


def _ensure_process_mask_patch() -> None:
    global _PROCESS_MASK_PATCHED, _ORIGINAL_PROCESS_MASK_NATIVE
    if _PROCESS_MASK_PATCHED:
        return
    try:
        from ultralytics.utils import ops as _ops
    except Exception:
        return

    _ORIGINAL_PROCESS_MASK_NATIVE = getattr(_ops, "process_mask_native", None)
    if _ORIGINAL_PROCESS_MASK_NATIVE is None:
        return

    def _wrapped_process_mask_native(protos, coeffs, boxes, shape, *args, **kwargs):
        if _DEFAULT_LEGACY_MASKS:
            return _ORIGINAL_PROCESS_MASK_NATIVE(protos, coeffs, boxes, shape, *args, **kwargs)

        proto_tensor = _prepare_proto_tensor(protos, _PROTO_UPSAMPLE_FACTOR)
        try:
            upsample = kwargs.get("upsample", True)
        except Exception:
            upsample = True
        try:
            soft_masks = _compute_native_soft_masks(proto_tensor, coeffs, boxes, shape, upsample=upsample)
            if soft_masks is not None:
                _MASK_PROB_CACHE.append(soft_masks.detach().cpu())
        except Exception:
            pass
        proto_arg = proto_tensor if proto_tensor is not None else protos
        return _ORIGINAL_PROCESS_MASK_NATIVE(proto_arg, coeffs, boxes, shape, *args, **kwargs)

    _ops.process_mask_native = _wrapped_process_mask_native
    _PROCESS_MASK_PATCHED = True


def _restore_process_mask_patch() -> None:
    global _PROCESS_MASK_PATCHED, _ORIGINAL_PROCESS_MASK_NATIVE
    if not _PROCESS_MASK_PATCHED:
        return
    try:
        from ultralytics.utils import ops as _ops
    except Exception:
        _PROCESS_MASK_PATCHED = False
        _ORIGINAL_PROCESS_MASK_NATIVE = None
        return

    if _ORIGINAL_PROCESS_MASK_NATIVE is not None:
        _ops.process_mask_native = _ORIGINAL_PROCESS_MASK_NATIVE

    _PROCESS_MASK_PATCHED = False
    _ORIGINAL_PROCESS_MASK_NATIVE = None


@contextmanager
def _process_mask_patch(proto_factor: int, legacy: bool):
    global _PROTO_UPSAMPLE_FACTOR, _DEFAULT_LEGACY_MASKS

    prev_factor = _PROTO_UPSAMPLE_FACTOR
    prev_legacy = _DEFAULT_LEGACY_MASKS
    pre_existing_patch = _PROCESS_MASK_PATCHED

    try:
        _PROTO_UPSAMPLE_FACTOR = proto_factor
        _DEFAULT_LEGACY_MASKS = legacy
        if not pre_existing_patch:
            _ensure_process_mask_patch()
        yield _PROCESS_MASK_PATCHED
    finally:
        if not pre_existing_patch:
            _restore_process_mask_patch()
        _PROTO_UPSAMPLE_FACTOR = prev_factor
        _DEFAULT_LEGACY_MASKS = prev_legacy
        _MASK_PROB_CACHE.clear()


def _pop_cached_prob_masks_any(roi_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if not _MASK_PROB_CACHE:
        return None
    tensor = _MASK_PROB_CACHE.popleft()
    if tensor is None:
        return None
    try:
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        prob = tensor.float().cpu().numpy().astype(np.float32, copy=False)
    except Exception:
        return None
    h, w = roi_shape
    if prob.shape[-2:] != (h, w):
        resized = [
            cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32, copy=False)
            for mask in prob
        ]
        prob = np.stack(resized, axis=0) if resized else np.empty((0, h, w), dtype=np.float32)
    return np.clip(prob, 0.0, 1.0)


def _repeat_to_rgb(batch: np.ndarray) -> List[np.ndarray]:
    if batch.ndim != 3:
        raise ValueError("ROI images expected shape (N, H, W)")
    return [np.repeat(img[..., None], 3, axis=2) for img in batch]


def _compute_mid_fraction(tensor: torch.Tensor) -> Optional[float]:
    if tensor is None or not torch.is_floating_point(tensor):
        return None
    if tensor.numel() == 0:
        return None
    try:
        mid = ((tensor > 1e-6) & (tensor < 1 - 1e-6)).float().mean()
        return float(mid.item())
    except Exception:
        return None


def _resize_prob_masks(prob_masks: np.ndarray, roi_shape: Tuple[int, int]) -> np.ndarray:
    h, w = roi_shape
    if prob_masks.ndim == 2:
        prob_masks = prob_masks[None, ...]
    if prob_masks.shape[-2:] == (h, w):
        return prob_masks.astype(np.float32, copy=False)
    resized_masks = [
        cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32, copy=False)
        for mask in prob_masks
    ]
    if not resized_masks:
        return np.empty((0, h, w), dtype=np.float32)
    return np.stack(resized_masks, axis=0)


def _reconstruct_masks_from_protos(result, roi_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    masks_obj = getattr(result, "masks", None)
    boxes_obj = getattr(result, "boxes", None)
    if masks_obj is None or boxes_obj is None:
        return None

    proto_raw = (
        getattr(masks_obj, "proto", None)
        or getattr(masks_obj, "protos", None)
        or getattr(result, "proto", None)
        or getattr(result, "protos", None)
    )
    proto_tensor = _prepare_proto_tensor(proto_raw, _PROTO_UPSAMPLE_FACTOR)
    if proto_tensor is None:
        return None

    coeffs = getattr(masks_obj, "weights", None)
    if coeffs is None:
        boxes_data = getattr(boxes_obj, "data", None)
        if boxes_data is not None and boxes_data.shape[1] > 6:
            coeffs = boxes_data[:, 6:]
    if coeffs is None:
        return None

    xyxy = getattr(boxes_obj, "xyxy", None)
    if xyxy is None:
        boxes_data = getattr(boxes_obj, "data", None)
        if boxes_data is not None and boxes_data.shape[1] >= 4:
            xyxy = boxes_data[:, :4]
    if xyxy is None:
        return None

    coeffs_tensor = coeffs if torch.is_tensor(coeffs) else torch.as_tensor(coeffs)
    xyxy_tensor = xyxy if torch.is_tensor(xyxy) else torch.as_tensor(xyxy)

    target_device = proto_tensor.device if hasattr(proto_tensor, "device") else coeffs_tensor.device
    proto_tensor = proto_tensor.to(target_device)
    coeffs_tensor = coeffs_tensor.to(target_device)
    xyxy_tensor = xyxy_tensor.to(target_device)

    process_fn = _ORIGINAL_PROCESS_MASK_NATIVE
    if process_fn is None:
        try:
            from ultralytics.utils.ops import process_mask_native as process_fn
        except Exception:
            return None
    try:
        masks = process_fn(proto_tensor, coeffs_tensor, xyxy_tensor, roi_shape)
    except Exception:
        return None
    if masks is None:
        return None
    masks = masks.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.clip(masks, 0.0, 1.0)


def _resolve_soft_masks(
    result,
    roi_shape: Tuple[int, int],
    *,
    expect_soft: bool,
    allow_fallback: bool,
) -> Tuple[np.ndarray, float, str, bool, bool, int, int, int, int]:
    h, w = roi_shape
    masks_obj = getattr(result, "masks", None)
    if masks_obj is None or masks_obj.data is None:
        return np.empty((0, h, w), dtype=np.float32), 0.0, "none", False, False, 0, 0, 0, 0

    tensor = masks_obj.data
    np_masks = tensor.detach().float().cpu().numpy()
    data_masks = _resize_prob_masks(np_masks, roi_shape)
    n_det_raw = int(data_masks.shape[0])

    boxes_obj = getattr(result, "boxes", None)
    n_boxes = 0
    if boxes_obj is not None:
        xyxy = getattr(boxes_obj, "xyxy", None)
        if xyxy is not None:
            n_boxes = int(xyxy.shape[0])
        else:
            boxes_data = getattr(boxes_obj, "data", None)
            if boxes_data is not None:
                n_boxes = int(boxes_data.shape[0])

    queue_len = len(_MASK_PROB_CACHE)
    hook_masks = _pop_cached_prob_masks_any(roi_shape) if queue_len else None
    n_cached = int(hook_masks.shape[0]) if hook_masks is not None and hook_masks.size else 0

    prob_masks: Optional[np.ndarray] = None
    source = "none"
    used_cache = False
    used_reconstruct = False

    reconstructed = _reconstruct_masks_from_protos(result, roi_shape)
    if reconstructed is not None and reconstructed.size:
        prob_masks = reconstructed
        source = "reconstruct"
        used_reconstruct = True
    elif hook_masks is not None and hook_masks.size:
        prob_masks = hook_masks
        source = "cache"
        used_cache = True
    elif n_det_raw > 0:
        prob_masks = data_masks
        source = "masks_data"
    else:
        prob_masks = np.empty((0, h, w), dtype=np.float32)

    prob_masks = np.clip(prob_masks, 0.0, 1.0)
    mid_fraction = float(((prob_masks > 1e-6) & (prob_masks < 1 - 1e-6)).mean()) if prob_masks.size else 0.0

    if (
        source == "masks_data"
        and allow_fallback
        and expect_soft
        and mid_fraction <= 1e-6
        and hook_masks is not None
        and hook_masks.size
    ):
        prob_masks = np.clip(hook_masks, 0.0, 1.0)
        source = "cache"
        used_cache = True
        mid_fraction = float(((prob_masks > 1e-6) & (prob_masks < 1 - 1e-6)).mean()) if prob_masks.size else 0.0

    return prob_masks, mid_fraction, source, used_cache, used_reconstruct, n_det_raw, n_cached, queue_len, n_boxes


def _candidate_from_mask(mask: np.ndarray, prob_mask: Optional[np.ndarray] = None) -> Optional[EyeCandidate]:
    if mask.sum() <= 0:
        return None
    props = measure.regionprops(mask.astype(np.uint8))
    if not props:
        return None
    region = max(props, key=lambda p: p.area)
    centroid = (float(region.centroid[1]), float(region.centroid[0]))
    ellipse_row = np.array(
        [
            centroid[0],
            centroid[1],
            float(region.major_axis_length),
            float(region.minor_axis_length),
            float(math.degrees(region.orientation)),
        ],
        dtype=np.float32,
    )
    contour = measure.find_contours(mask.astype(float), 0.5)
    contour_xy: Optional[np.ndarray] = None
    if contour:
        best = max(contour, key=lambda arr: arr.shape[0])
        contour_xy = best[:, ::-1].astype(np.float32)
    return EyeCandidate(
        mask=mask,
        prob_mask=prob_mask,
        centroid_xy=centroid,
        ellipse_row=ellipse_row,
        contour_xy=contour_xy,
        area=float(region.area),
    )


def _soft_ellipse_from_prob(prob_mask: Optional[np.ndarray], component_mask: np.ndarray) -> Optional[np.ndarray]:
    if prob_mask is None:
        return None
    if prob_mask.shape != component_mask.shape:
        return None

    weights = prob_mask.astype(np.float32, copy=False) * component_mask.astype(np.float32, copy=False)
    weight_sum = float(weights.sum())
    if weight_sum <= 1e-6 or not math.isfinite(weight_sum):
        return None

    yy, xx = np.indices(weights.shape, dtype=np.float32)
    x_mean = float((weights * xx).sum() / weight_sum)
    y_mean = float((weights * yy).sum() / weight_sum)

    x_diff = xx - x_mean
    y_diff = yy - y_mean

    xx_moment = float((weights * x_diff * x_diff).sum() / weight_sum)
    yy_moment = float((weights * y_diff * y_diff).sum() / weight_sum)
    xy_moment = float((weights * x_diff * y_diff).sum() / weight_sum)

    cov = np.array([[xx_moment, xy_moment], [xy_moment, yy_moment]], dtype=np.float32)
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return None

    eigvals = np.clip(eigvals, 0.0, None)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    eps = 1e-12
    if not np.isfinite(eigvals[0]) or not np.isfinite(eigvals[1]) or eigvals[0] < eps:
        return None

    major_length = float(4.0 * math.sqrt(eigvals[0]))
    minor_length = float(4.0 * math.sqrt(max(eps, eigvals[1])))

    orient_rad = math.atan2(float(eigvecs[1, 0]), float(eigvecs[0, 0]))
    orient_deg = float(math.degrees(orient_rad))
    if orient_deg < -90.0:
        orient_deg += 180.0
    elif orient_deg > 90.0:
        orient_deg -= 180.0

    return np.array([x_mean, y_mean, major_length, minor_length, orient_deg], dtype=np.float32)


def _binarize_adaptive(
    prob_mask: np.ndarray,
    global_floor: float,
    adaptive_scale: float,
    adaptive_cap: float,
) -> np.ndarray:
    pmax = float(prob_mask.max())
    if pmax <= 1e-6:
        return np.zeros_like(prob_mask, dtype=np.uint8)

    floor = max(0.0, float(global_floor))
    scale = max(0.0, float(adaptive_scale))
    cap = max(0.0, min(1.0, float(adaptive_cap)))

    adaptive = scale * pmax
    if cap > 0.0:
        adaptive = min(cap, adaptive)
    threshold = max(floor, adaptive)
    binary = (prob_mask >= threshold).astype(np.uint8)

    if not binary.any():
        return binary

    num_labels, labels = cv2.connectedComponents(binary, connectivity=8)
    if num_labels > 2:
        counts = np.bincount(labels.ravel())
        keep_label = int(np.argmax(counts[1:])) + 1
        binary = (labels == keep_label).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    coords = np.argwhere(binary)
    bounding_mask: Optional[np.ndarray] = None
    if coords.size:
        min_r, min_c = coords.min(axis=0)
        max_r, max_c = coords.max(axis=0)
        bounding_mask = np.zeros_like(binary, dtype=bool)
        bounding_mask[min_r : max_r + 1, min_c : max_c + 1] = True

    inv = (1 - binary).astype(np.uint8)
    if bounding_mask is not None:
        inv[~bounding_mask] = 0
    num_holes, hole_labels = cv2.connectedComponents(inv, connectivity=8)
    for label_id in range(1, num_holes):
        hole_mask = hole_labels == label_id
        if bounding_mask is not None and not np.any(hole_mask):
            continue
        hole_area = int(np.sum(hole_mask))
        if hole_area <= 10:
            binary[hole_mask] = 1

    return binary


def _extract_candidates(
    prob_masks: np.ndarray,
    mask_threshold: float,
    adaptive_scale: float,
    adaptive_cap: float,
) -> List[EyeCandidate]:
    if prob_masks.size == 0:
        return []

    candidate_list: List[EyeCandidate] = []
    for mask in prob_masks:
        bin_mask = _binarize_adaptive(
            mask.astype(np.float32, copy=False),
            mask_threshold,
            adaptive_scale,
            adaptive_cap,
        )
        candidate = _candidate_from_mask(bin_mask, mask.astype(np.float32, copy=False))
        if candidate is not None:
            candidate_list.append(candidate)

    candidate_list.sort(key=lambda c: (c.area, c.centroid_xy[0]), reverse=True)
    return candidate_list[:2]


def _select_eye_candidates(candidates: Sequence[EyeCandidate], max_eyes: int = 2) -> List[Optional[EyeCandidate]]:
    """Return up to ``max_eyes`` candidates, padding with ``None`` if needed."""
    if max_eyes <= 0:
        return []
    ordered = list(candidates)
    selected: List[Optional[EyeCandidate]] = ordered[:max_eyes]
    while len(selected) < max_eyes:
        selected.append(None)
    return selected


def segment_eye_masks_yolo(
    zarr_path: str,
    model_path: str,
    *,
    run_name: Optional[str] = None,
    crop_run: Optional[str] = None,
    keypoints_run: Optional[str] = None,
    batch_size: int = 128,
    device: Optional[str] = None,
    imgsz: Optional[int] = None,
    conf: float = 0.05,
    iou: float = 0.5,
    max_det: int = 2,
    mask_threshold: float = 0.05,
    adaptive_scale: float = 0.6,
    adaptive_cap: float = 0.6,
    use_retina_masks: bool = True,
    proto_upsample_factor: int = 2,
    legacy_masks: bool = False,
    verbose: bool = False,
    console: Optional[Console] = None,
) -> str:
    """Run a YOLO segmentation model to generate binary and probability eye masks.

    The `mask_threshold` argument acts as the global floor for adaptive binarization; the
    effective threshold per instance is `max(mask_threshold, min(adaptive_cap, adaptive_scale * pmax))`.
    """

    from ultralytics import YOLO, __version__ as ultralytics_version

    console = console or Console()
    console.rule("[bold cyan]YOLO Eye Segmentation[/bold cyan]")

    zarr_path = Path(zarr_path)
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr path not found: {zarr_path}")

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    model = YOLO(str(model_path))
    if device:
        model.to(device)
    try:
        model_device = str(next(model.model.parameters()).device)
    except (AttributeError, StopIteration):
        model_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_path_resolved = model_path.resolve()

    root = zarr.open(str(zarr_path), mode="a")
    if "crop_runs" not in root:
        raise ValueError("Zarr archive missing crop_runs; run cropping first")

    crop_run_name = crop_run or root["crop_runs"].attrs.get("latest")
    if crop_run_name is None:
        raise ValueError("No crop run available; cannot perform eye segmentation")
    crop_group = root[f"crop_runs/{crop_run_name}"]
    roi_images = crop_group["roi_images"]

    global _PROTO_UPSAMPLE_FACTOR, _DEFAULT_LEGACY_MASKS
    proto_factor = max(1, int(proto_upsample_factor))
    legacy_flag = bool(legacy_masks)

    total_rois = int(roi_images.shape[0])
    if total_rois == 0:
        console.print("[yellow]No ROIs available; nothing to segment[/yellow]")
        _PROTO_UPSAMPLE_FACTOR = 1
        return ""

    roi_h, roi_w = int(roi_images.shape[1]), int(roi_images.shape[2])
    if imgsz is None:
        imgsz_resolved = int(max(roi_h, roi_w))
        if verbose:
            console.print(
                f"[dim]imgsz not provided; using ROI max dimension {imgsz_resolved}[/dim]"
            )
    else:
        imgsz_resolved = int(imgsz)

    _validate_input_row_alignment(
        crop_group=crop_group,
        crop_run_name=str(crop_run_name),
        total_rois=total_rois,
    )

    run_group, resolved_run_name = _prepare_run_group(root, run_name, console)

    def _copy_metadata_array(array_name: str) -> None:
        if array_name not in crop_group:
            console.print(f"[yellow]Crop run missing '{array_name}'; new eye mask run will skip it.[/yellow]")
            return
        source = crop_group[array_name]
        data = source[:]
        source_chunks = getattr(source, "chunks", None)
        if not source_chunks:
            source_chunks = tuple(max(1, min(dim, 1024)) for dim in data.shape)
        else:
            chunk_list: List[int] = []
            for axis, dim in enumerate(data.shape):
                chunk_val = source_chunks[axis] if axis < len(source_chunks) else source_chunks[-1]
                chunk_list.append(int(max(1, min(dim, chunk_val))))
            source_chunks = tuple(chunk_list)
        if array_name in run_group:
            del run_group[array_name]
        run_group.create_array(
            array_name,
            data=data,
            chunks=source_chunks,
            overwrite=True,
        )

    _copy_metadata_array("frame_indices")
    _copy_metadata_array("detection_indices")
    _copy_metadata_array("frame_counts")

    masks = np.zeros((total_rois, 2, roi_h, roi_w), dtype=np.uint8)
    mask_probs = np.zeros((total_rois, 2, roi_h, roi_w), dtype=np.float16)
    ellipse_params = np.full((total_rois, 2, 5), np.nan, dtype=np.float32)
    ellipse_params_soft = np.full((total_rois, 2, 5), np.nan, dtype=np.float32)
    ellipse_success = np.zeros((total_rois, 2), dtype=bool)
    eye_separation = np.full((total_rois,), np.nan, dtype=np.float32)

    contour_ptr = np.full((total_rois, 2), -1, dtype=np.int64)
    contour_len = np.zeros((total_rois, 2), dtype=np.int32)
    contour_points: List[List[np.ndarray]] = [[], []]
    contour_totals = [0, 0]

    successful_pairs = 0

    stage_start = time.perf_counter()
    _MASK_PROB_CACHE.clear()

    with _process_mask_patch(proto_factor, legacy_flag) as patch_active:
        if not patch_active and not legacy_flag and verbose:
            console.print(
                "[yellow]Soft mask hook unavailable; probability masks may remain Ultralytics defaults[/yellow]"
            )

        timer = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeRemainingColumn(),
            console=console,
        )

        printed_prob_stats = False
        prototype_fallback_logged = False
        prototype_fallback_used = False
        native_cache_logged = False
        native_cache_used = False
        fallback_failure_logged = False
        retina_warning_logged = False
        soft_rows_written = 0
        pmax_samples: List[float] = []
        mid_fraction_samples: List[float] = []

        model_kwargs: Dict[str, object] = {
            "imgsz": imgsz_resolved,
            "conf": conf,
            "iou": iou,
            "max_det": max_det,
            "device": device,
            "verbose": verbose,
        }
        if not legacy_flag and use_retina_masks is not None:
            model_kwargs["retina_masks"] = use_retina_masks
        retina_flag_supported = True

        with timer:
            task_id = timer.add_task("[cyan]Running YOLO segmentation...[/cyan]", total=total_rois)
            for start in range(0, total_rois, batch_size):
                end = min(start + batch_size, total_rois)
                batch = np.asarray(roi_images[start:end])
                rgb_batch = _repeat_to_rgb(batch)

                try:
                    results = model(rgb_batch, **model_kwargs)
                except TypeError as exc:
                    if "retina_masks" in str(exc) and retina_flag_supported and "retina_masks" in model_kwargs:
                        retina_flag_supported = False
                        model_kwargs.pop("retina_masks", None)
                        if not retina_warning_logged:
                            console.print(
                                "[yellow]Retina mask flag not supported by this Ultralytics build; falling back to default masks[/yellow]"
                            )
                            retina_warning_logged = True
                        results = model(rgb_batch, **model_kwargs)
                    else:
                        raise

                retina_active = bool(model_kwargs.get("retina_masks", False)) and retina_flag_supported and not _DEFAULT_LEGACY_MASKS

                for idx, result in enumerate(results):
                    global_idx = start + idx
                    if _DEFAULT_LEGACY_MASKS:
                        (
                            prob_masks,
                            mid_fraction,
                            source_kind,
                            used_cache,
                            used_reconstruct,
                            n_det_raw,
                            n_cached,
                            cache_qlen,
                            n_boxes,
                        ) = _resolve_soft_masks(
                            result,
                            (roi_h, roi_w),
                            expect_soft=False,
                            allow_fallback=False,
                        )
                    else:
                        (
                            prob_masks,
                            mid_fraction,
                            source_kind,
                            used_cache,
                            used_reconstruct,
                            n_det_raw,
                            n_cached,
                            cache_qlen,
                            n_boxes,
                        ) = _resolve_soft_masks(
                            result,
                            (roi_h, roi_w),
                            expect_soft=retina_active,
                            allow_fallback=True,
                        )
                    n_det = int(prob_masks.shape[0]) if prob_masks.ndim >= 1 else 0
                    if prob_masks.size:
                        pmax_samples.append(float(prob_masks.max()))
                    else:
                        pmax_samples.append(0.0)
                    mid_fraction_samples.append(float(mid_fraction))
                    if verbose and not printed_prob_stats and prob_masks.size > 0:
                        per_max = [float(pm.max()) for pm in prob_masks]
                        per_mean = [float(pm.mean()) for pm in prob_masks]
                        mid = float(((prob_masks > 1e-6) & (prob_masks < 1 - 1e-6)).mean())
                        nz_mid: List[float] = []
                        for pm in prob_masks:
                            nz = pm[pm > 0]
                            nz_mid.append(
                                float(((nz > 1e-6) & (nz < 1 - 1e-6)).mean()) if nz.size else 0.0
                            )
                        console.print(
                            "[cyan]mask probability stats[/cyan]: "
                            f"mid_fraction(all)={mid:.6f} | "
                            f"per_inst_max={[round(x, 4) for x in per_max[:6]]} | "
                            f"per_inst_mean={[round(x, 5) for x in per_mean[:6]]} | "
                            f"nz_mid={[round(x, 3) for x in nz_mid[:6]]}"
                        )
                        console.print(
                            f"[cyan]path[/cyan]: retina_active={retina_active}, source={source_kind}, used_cache={used_cache}, "
                            f"used_reconstruct={used_reconstruct}, cache_qlen={cache_qlen}, n_boxes={n_boxes}, "
                            f"n_det={n_det_raw}, n_cached={n_cached}, n_final={n_det}"
                        )
                        printed_prob_stats = True
                    if (
                        not used_cache
                        and not used_reconstruct
                        and not _DEFAULT_LEGACY_MASKS
                        and mid_fraction <= 1e-6
                    ):
                        rebuilt = _reconstruct_masks_from_protos(result, (roi_h, roi_w))
                        if rebuilt is not None and rebuilt.size:
                            prob_masks = rebuilt
                            used_reconstruct = True
                            source_kind = "reconstruct"
                            mid_fraction = float(((prob_masks > 1e-6) & (prob_masks < 1 - 1e-6)).mean()) if prob_masks.size else 0.0
                    if used_reconstruct:
                        prototype_fallback_used = True
                    if verbose and used_reconstruct and not prototype_fallback_logged:
                        console.print("[yellow]Prototype-based mask reconstruction enabled to recover soft probabilities[/yellow]")
                        prototype_fallback_logged = True
                    if used_cache:
                        native_cache_used = True
                    if verbose and used_cache and not native_cache_logged:
                        console.print("[cyan]Captured soft masks directly from Ultralytics process_mask_native hook[/cyan]")
                        native_cache_logged = True
                    if (
                        verbose
                        and not used_reconstruct
                        and not used_cache
                        and mid_fraction <= 1e-6
                        and not fallback_failure_logged
                    ):
                        if retina_active:
                            console.print(
                                "[yellow]Retina masks appear binary and prototype reconstruction was unavailable; saved masks may be binary[/yellow]"
                            )
                        else:
                            console.print(
                                "[yellow]Prototype reconstruction could not be applied; saved probability masks may remain binary[/yellow]"
                            )
                        fallback_failure_logged = True

                    candidates = _extract_candidates(
                        prob_masks,
                        mask_threshold,
                        adaptive_scale,
                        adaptive_cap,
                    )
                    candidate_slots = _select_eye_candidates(candidates)

                    centroids: List[Optional[Tuple[float, float]]] = [None, None]

                    for eye_idx, candidate in enumerate(candidate_slots):
                        if candidate is None:
                            continue
                        bin_mask = candidate.mask.astype(np.uint8)
                        masks[global_idx, eye_idx] = bin_mask
                        if candidate.prob_mask is not None:
                            mask_probs[global_idx, eye_idx] = candidate.prob_mask.astype(np.float16, copy=False)
                        ellipse_params[global_idx, eye_idx] = candidate.ellipse_row
                        ellipse_success[global_idx, eye_idx] = True
                        soft_row = _soft_ellipse_from_prob(candidate.prob_mask, bin_mask)
                        if soft_row is not None:
                            ellipse_params_soft[global_idx, eye_idx] = soft_row
                            soft_rows_written += 1
                        centroids[eye_idx] = candidate.centroid_xy
                        if candidate.contour_xy is not None:
                            contour = candidate.contour_xy
                            contour_ptr[global_idx, eye_idx] = contour_totals[eye_idx]
                            contour_len[global_idx, eye_idx] = contour.shape[0]
                            contour_points[eye_idx].append(contour)
                            contour_totals[eye_idx] += contour.shape[0]

                    if all(point is not None for point in centroids):
                        left_pt = centroids[0]
                        right_pt = centroids[1]
                        separation = math.hypot(left_pt[0] - right_pt[0], left_pt[1] - right_pt[1])
                        eye_separation[global_idx] = float(separation)
                        successful_pairs += 1

                timer.update(task_id, advance=end - start)

    _MASK_PROB_CACHE.clear()

    duration = time.perf_counter() - stage_start

    pmax_arr = np.asarray(pmax_samples, dtype=np.float32) if pmax_samples else np.array([], dtype=np.float32)
    mid_arr = np.asarray(mid_fraction_samples, dtype=np.float32) if mid_fraction_samples else np.array([], dtype=np.float32)
    pmax_mean = float(pmax_arr.mean()) if pmax_arr.size else float("nan")
    pmax_p95 = float(np.percentile(pmax_arr, 95)) if pmax_arr.size else float("nan")
    mid_frac_mean = float(mid_arr.mean()) if mid_arr.size else float("nan")

    contour_concat = [
        (
            np.concatenate(contour_points[eye_idx], axis=0).astype(np.float32)
            if contour_points[eye_idx]
            else np.zeros((0, 2), dtype=np.float32)
        )
        for eye_idx in range(2)
    ]
    contour_store = [
        concat if concat.size > 0 else np.zeros((1, 2), dtype=np.float32)
        for concat in contour_concat
    ]

    chunk_rois = min(512, total_rois) if total_rois > 0 else 1
    run_group.create_array(
        "masks_roi",
        data=masks,
        chunks=(chunk_rois, 2, roi_h, roi_w),
        overwrite=True,
    )
    run_group.create_array(
        "mask_probs_roi",
        data=mask_probs,
        chunks=(chunk_rois, 2, roi_h, roi_w),
        overwrite=True,
    )
    run_group.create_array(
        "ellipse_params",
        data=ellipse_params,
        chunks=(min(1024, total_rois), 2, 5),
        overwrite=True,
    )
    if soft_rows_written > 0:
        run_group.create_array(
            "ellipse_params_soft",
            data=ellipse_params_soft,
            chunks=(min(1024, total_rois), 2, 5),
            overwrite=True,
        )
    run_group.create_array(
        "ellipse_success",
        data=ellipse_success,
        chunks=(min(1024, total_rois), 2),
        overwrite=True,
    )
    run_group.create_array(
        "eye_separation",
        data=eye_separation,
        chunks=(min(1024, total_rois),),
        overwrite=True,
    )
    run_group.create_array(
        "contour_ptr",
        data=contour_ptr,
        chunks=(min(1024, total_rois), 2),
        overwrite=True,
    )
    run_group.create_array(
        "contour_len",
        data=contour_len,
        chunks=(min(1024, total_rois), 2),
        overwrite=True,
    )
    run_group.create_array(
        "contours_eye0",
        data=contour_store[0],
        chunks=(max(1, min(4096, contour_store[0].shape[0])), 2),
        overwrite=True,
    )
    run_group.create_array(
        "contours_eye1",
        data=contour_store[1],
        chunks=(max(1, min(4096, contour_store[1].shape[0])), 2),
        overwrite=True,
    )

    git_info = get_git_info()
    env_info = get_environment_info(
        include_all_packages=False,
        disk_path=str(zarr_path),
        collect_ip=False,
        capture_env_vars=False,
    )
    platform_info = env_info.get("platform", {})
    total_successful_eyes = int(ellipse_success.sum())
    pair_rate = float(successful_pairs / total_rois) if total_rois > 0 else float("nan")

    resolved_keypoints_run, resolved_keypoint_group = _resolve_keypoint_lineage(
        root,
        crop_group,
        keypoints_run,
    )

    attrs_payload = {
        "method": "yolo_eye_segmentation",
        "model_path": str(model_path_resolved),
        "model_device": model_device,
        "ultralytics_version": ultralytics_version,
        "config": {
            "batch_size": batch_size,
            "imgsz": imgsz_resolved,
            "conf": conf,
            "iou": iou,
            "max_det": max_det,
            "mask_threshold": mask_threshold,
            "adaptive_scale": adaptive_scale,
            "adaptive_cap": adaptive_cap,
            "use_retina_masks": use_retina_masks,
            "legacy_masks": bool(legacy_masks),
            "retina_masks_supported": retina_flag_supported,
            "prototype_fallback_used": prototype_fallback_used,
            "native_mask_hook_used": native_cache_used,
            "proto_upsample_factor": proto_factor,
        },
        "source_crop_run": crop_run_name,
        **build_source_keypoints_attrs(resolved_keypoints_run, include_legacy_alias=True),
        "total_rois": total_rois,
        "successful_eyes": total_successful_eyes,
        "successful_roi_pairs": int(successful_pairs),
        "successful_roi_pair_rate": pair_rate,
        "eye_labels": ["eye_0", "eye_1"],
        "git_commit": git_info.get("commit_hash", "unknown"),
        "git_branch": git_info.get("branch", "unknown"),
        "hostname": env_info["platform"].get("hostname", "unknown"),
        "ellipse_soft_available": bool(soft_rows_written > 0),
        "ellipse_soft_convention": {
            "lengths": "4*sqrt(eigvals) from prob-weighted covariance",
            "angle_ref": "deg CCW from +x, normalized to [-90, 90]",
        },
        "probability_stats": {
            "pmax_mean": pmax_mean,
            "pmax_p95": pmax_p95,
            "mid_fraction_mean": mid_frac_mean,
        },
        "duration_seconds": float(duration),
        "ellipse_angle_units": "degrees",
        "ellipse_angle_ref": "skimage major-axis orientation, deg CCW from +x",
    }
    if resolved_keypoint_group is not None:
        attrs_payload["source_keypoint_group"] = resolved_keypoint_group

    run_group.attrs.update(attrs_payload)
    provenance = {
        "stage": "eye_masks",
        "method": "yolo_eye_segmentation",
        "command": " ".join(sys.argv),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_info,
        "environment": env_info.get("environment"),
        "platform": {
            "hostname": platform_info.get("hostname"),
            "system": platform_info.get("system"),
            "release": platform_info.get("release"),
            "python_version": platform_info.get("python_version"),
            "machine": platform_info.get("machine"),
        },
        "parameters": run_group.attrs.get("config"),
        "inputs": {
            "source_crop_run": crop_run_name,
            "source_keypoints_run": resolved_keypoints_run,
            "source_keypoint_group": resolved_keypoint_group,
            "frame_source": crop_group.attrs.get("video_source_type", "zarr"),
            "source_video_path": crop_group.attrs.get("video_source_path"),
        },
        "artifacts": {
            "model_path": str(model_path_resolved),
            "model_device": model_device,
            "ultralytics_version": ultralytics_version,
        },
    }
    provenance = {k: v for k, v in provenance.items() if v is not None}
    run_group.attrs["provenance"] = provenance
    run_group.attrs["rejected_overlap"] = 0
    run_group.attrs["rejected_too_close"] = 0
    run_group.attrs["rejected_too_far"] = 0

    console.print(
        f"[green]✓[/green] Eye masks saved as [cyan]eye_masks_runs/{resolved_run_name}[/cyan] "
        f"({total_successful_eyes} successful eyes, {successful_pairs}/{total_rois} ROI pairs) in {duration:.1f}s"
    )

    _PROTO_UPSAMPLE_FACTOR = 1
    _DEFAULT_LEGACY_MASKS = False

    return resolved_run_name


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run YOLO-based eye segmentation on Palette ROI crops."
    )
    parser.add_argument("zarr_path", help="Path to the Palette Zarr archive.")
    parser.add_argument("model_path", help="Path to the YOLO segmentation weights (.pt).")
    parser.add_argument("--run-name", help="Optional name for the output run.")
    parser.add_argument("--crop-run", help="Crop run providing ROI images (default: latest).")
    parser.add_argument("--keypoints-run", help="Keypoint run that produced the ROIs (default: inferred).")
    parser.add_argument("--batch-size", type=int, default=128, help="Inference batch size (default: 128).")
    parser.add_argument("--device", help="Torch device to run on (e.g. cuda:0, cpu).")
    parser.add_argument("--imgsz", type=int, help="Override inference image size (defaults to ROI size).")
    parser.add_argument("--conf", type=float, default=0.05, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="YOLO IoU threshold for NMS.")
    parser.add_argument("--max-det", type=int, default=2, help="Maximum detections per ROI (default: 2).")
    parser.add_argument("--mask-threshold", type=float, default=0.05, help="Minimum probability when binarizing masks.")
    parser.add_argument("--adaptive-scale", type=float, default=0.6, help="Scale factor for adaptive thresholding.")
    parser.add_argument("--adaptive-cap", type=float, default=0.6, help="Cap for adaptive thresholding.")
    parser.add_argument("--proto-upsample-factor", type=int, default=2, help="Prototype upsample factor for soft masks.")
    parser.add_argument("--no-retina", action="store_true", help="Disable Ultralytics retina masks.")
    parser.add_argument("--legacy-masks", action="store_true", help="Force legacy mask extraction mode.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    console = Console()
    segment_eye_masks_yolo(
        zarr_path=args.zarr_path,
        model_path=args.model_path,
        run_name=args.run_name,
        crop_run=args.crop_run,
        keypoints_run=args.keypoints_run,
        batch_size=args.batch_size,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        mask_threshold=args.mask_threshold,
        adaptive_scale=args.adaptive_scale,
        adaptive_cap=args.adaptive_cap,
        use_retina_masks=not args.no_retina,
        proto_upsample_factor=args.proto_upsample_factor,
        legacy_masks=args.legacy_masks,
        verbose=args.verbose,
        console=console,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()


__all__ = ["segment_eye_masks_yolo"]
