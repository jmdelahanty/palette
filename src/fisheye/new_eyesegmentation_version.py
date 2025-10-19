"""YOLO-based eye segmentation for Palette ROI crops."""

from __future__ import annotations

import math
from collections import deque
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

from .shared.zarr.schema import get_run_group
from .utils.system import get_environment_info, get_git_info


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


_MASK_PROB_CACHE: Deque[torch.Tensor] = deque()
_PROCESS_MASK_PATCHED = False
_ORIGINAL_PROCESS_MASK_NATIVE = None
_PROTO_UPSAMPLE_FACTOR = 1


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


def _pop_cached_prob_masks(expected_count: int, roi_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    if not _MASK_PROB_CACHE:
        return None
    tensor = _MASK_PROB_CACHE.popleft()
    if tensor is None:
        return None
    try:
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        prob = tensor.float().numpy()
    except Exception:
        return None
    if prob.shape[0] != expected_count:
        return None
    prob = prob.astype(np.float32, copy=False)
    if roi_shape and prob.shape[-2:] != tuple(roi_shape):
        prob = _resize_prob_masks(prob, roi_shape)
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
) -> Tuple[np.ndarray, Optional[float], bool, bool]:
    masks_obj = getattr(result, "masks", None)
    if masks_obj is None or masks_obj.data is None:
        h, w = roi_shape
        return np.empty((0, h, w), dtype=np.float32), None, False, False

    tensor = masks_obj.data
    mid_fraction = _compute_mid_fraction(tensor)
    np_masks = tensor.detach().float().cpu().numpy()
    prob_masks = _resize_prob_masks(np_masks, roi_shape)

    cache_used = False
    cached = _pop_cached_prob_masks(prob_masks.shape[0], roi_shape)
    if cached is not None and cached.shape[0] == prob_masks.shape[0]:
        prob_masks = cached
        cache_used = True
        mid_fraction = float(((prob_masks > 1e-6) & (prob_masks < 1 - 1e-6)).mean()) if prob_masks.size else 0.0

    fallback_needed = allow_fallback and (
        mid_fraction is None or (expect_soft and (mid_fraction <= 1e-6)) or (not expect_soft)
    )
    used_fallback = False
    if fallback_needed and not cache_used:
        rebuilt = _reconstruct_masks_from_protos(result, roi_shape)
        if rebuilt is not None:
            prob_masks = rebuilt
            used_fallback = True
            mid_fraction = float(((prob_masks > 1e-6) & (prob_masks < 1 - 1e-6)).mean()) if prob_masks.size else 0.0

    if mid_fraction is None and prob_masks.size:
        mid_fraction = float(((prob_masks > 1e-6) & (prob_masks < 1 - 1e-6)).mean())

    return np.clip(prob_masks, 0.0, 1.0), mid_fraction, used_fallback, cache_used


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


def _extract_candidates(
    prob_masks: np.ndarray,
    mask_threshold: float,
) -> List[EyeCandidate]:
    if prob_masks.size == 0:
        return []

    candidate_list: List[EyeCandidate] = []
    for mask in prob_masks:
        binary = mask >= mask_threshold
        candidate = _candidate_from_mask(binary.astype(np.uint8), mask.astype(np.float32, copy=False))
        if candidate is not None:
            candidate_list.append(candidate)

    candidate_list.sort(key=lambda c: (c.area, c.centroid_xy[0]), reverse=True)
    return candidate_list[:2]


def _assign_left_right(candidates: Sequence[EyeCandidate]) -> List[Optional[EyeCandidate]]:
    if not candidates:
        return [None, None]
    ordered = sorted(candidates, key=lambda c: c.centroid_xy[0])
    result: List[Optional[EyeCandidate]] = [None, None]
    if ordered:
        result[0] = ordered[0]
    if len(ordered) > 1:
        result[1] = ordered[1]
    return result


def segment_eye_masks_yolo(
    zarr_path: str,
    model_path: str,
    *,
    run_name: Optional[str] = None,
    crop_run: Optional[str] = None,
    batch_size: int = 128,
    device: Optional[str] = None,
    imgsz: Optional[int] = None,
    conf: float = 0.25,
    iou: float = 0.5,
    max_det: int = 4,
    mask_threshold: float = 0.5,
    use_retina_masks: bool = True,
    proto_upsample_factor: int = 1,
    verbose: bool = False,
    console: Optional[Console] = None,
) -> str:
    """Run a YOLO segmentation model to generate binary and probability eye masks."""

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

    global _PROTO_UPSAMPLE_FACTOR
    proto_factor = max(1, int(proto_upsample_factor))
    _PROTO_UPSAMPLE_FACTOR = proto_factor

    total_rois = int(roi_images.shape[0])
    if total_rois == 0:
        console.print("[yellow]No ROIs available; nothing to segment[/yellow]")
        _PROTO_UPSAMPLE_FACTOR = 1
        return ""

    roi_h, roi_w = int(roi_images.shape[1]), int(roi_images.shape[2])

    run_group, resolved_run_name = _prepare_run_group(root, run_name, console)

    masks = np.zeros((total_rois, 2, roi_h, roi_w), dtype=np.uint8)
    mask_probs = np.zeros((total_rois, 2, roi_h, roi_w), dtype=np.float16)
    ellipse_params = np.full((total_rois, 2, 5), np.nan, dtype=np.float32)
    ellipse_success = np.zeros((total_rois, 2), dtype=bool)
    feret_axes_major = np.full((total_rois, 2, 4), np.nan, dtype=np.float32)
    feret_axes_minor = np.full((total_rois, 2, 4), np.nan, dtype=np.float32)
    feret_roundness = np.full((total_rois, 2), np.nan, dtype=np.float32)
    eye_separation = np.full((total_rois,), np.nan, dtype=np.float32)

    left_ptr = np.full((total_rois,), -1, dtype=np.int64)
    right_ptr = np.full((total_rois,), -1, dtype=np.int64)
    left_len = np.zeros((total_rois,), dtype=np.int32)
    right_len = np.zeros((total_rois,), dtype=np.int32)
    left_points: List[np.ndarray] = []
    right_points: List[np.ndarray] = []
    left_total = 0
    right_total = 0

    successful_pairs = 0

    _ensure_process_mask_patch()
    _MASK_PROB_CACHE.clear()

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

    model_kwargs: Dict[str, object] = {
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "max_det": max_det,
        "device": device,
        "verbose": verbose,
    }
    if use_retina_masks is not None:
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

            retina_active = bool(model_kwargs.get("retina_masks", False)) and retina_flag_supported

            for idx, result in enumerate(results):
                global_idx = start + idx
                prob_masks, mid_fraction, used_fallback, used_cache = _resolve_soft_masks(
                    result,
                    (roi_h, roi_w),
                    expect_soft=retina_active,
                    allow_fallback=True,
                )
                if verbose and not printed_prob_stats and prob_masks.size > 0:
                    min_val = float(prob_masks.min())
                    max_val = float(prob_masks.max())
                    console.print(
                        "[cyan]mask probability stats[/cyan]: "
                        f"dtype=float32 min={min_val:.6f} max={max_val:.6f} mid_fraction={mid_fraction or 0.0:.6f}"
                    )
                    printed_prob_stats = True
                if used_fallback:
                    prototype_fallback_used = True
                if verbose and used_fallback and not prototype_fallback_logged:
                    console.print("[yellow]Prototype-based mask reconstruction enabled to recover soft probabilities[/yellow]")
                    prototype_fallback_logged = True
                if used_cache:
                    native_cache_used = True
                if verbose and used_cache and not native_cache_logged:
                    console.print("[cyan]Captured soft masks directly from Ultralytics process_mask_native hook[/cyan]")
                    native_cache_logged = True
                if (
                    verbose
                    and not used_fallback
                    and not used_cache
                    and mid_fraction is not None
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

                candidates = _extract_candidates(prob_masks, mask_threshold)
                left_right = _assign_left_right(candidates)

                centroids: List[Optional[Tuple[float, float]]] = [None, None]

                for eye_idx, candidate in enumerate(left_right):
                    if candidate is None:
                        continue
                    bin_mask = candidate.mask.astype(np.uint8)
                    masks[global_idx, eye_idx] = bin_mask
                    if candidate.prob_mask is not None:
                        mask_probs[global_idx, eye_idx] = candidate.prob_mask.astype(np.float16, copy=False)
                    ellipse_params[global_idx, eye_idx] = candidate.ellipse_row
                    ellipse_success[global_idx, eye_idx] = True
                    centroids[eye_idx] = candidate.centroid_xy
                    if candidate.contour_xy is not None:
                        contour = candidate.contour_xy
                        if eye_idx == 0:
                            left_ptr[global_idx] = left_total
                            left_len[global_idx] = contour.shape[0]
                            left_points.append(contour)
                            left_total += contour.shape[0]
                        else:
                            right_ptr[global_idx] = right_total
                            right_len[global_idx] = contour.shape[0]
                            right_points.append(contour)
                            right_total += contour.shape[0]

                if all(point is not None for point in centroids):
                    left_pt = centroids[0]
                    right_pt = centroids[1]
                    separation = math.hypot(left_pt[0] - right_pt[0], left_pt[1] - right_pt[1])
                    eye_separation[global_idx] = float(separation)
                    successful_pairs += 1

            timer.update(task_id, advance=end - start)

    _MASK_PROB_CACHE.clear()

    left_concat = (
        np.concatenate(left_points, axis=0).astype(np.float32) if left_points else np.zeros((0, 2), dtype=np.float32)
    )
    right_concat = (
        np.concatenate(right_points, axis=0).astype(np.float32) if right_points else np.zeros((0, 2), dtype=np.float32)
    )
    left_store = left_concat if left_concat.size > 0 else np.zeros((1, 2), dtype=np.float32)
    right_store = right_concat if right_concat.size > 0 else np.zeros((1, 2), dtype=np.float32)

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
    run_group.create_array(
        "ellipse_success",
        data=ellipse_success,
        chunks=(min(1024, total_rois), 2),
        overwrite=True,
    )
    run_group.create_array(
        "feret_axes_major",
        data=feret_axes_major,
        chunks=(min(1024, total_rois), 2, 4),
        overwrite=True,
    )
    run_group.create_array(
        "feret_axes_minor",
        data=feret_axes_minor,
        chunks=(min(1024, total_rois), 2, 4),
        overwrite=True,
    )
    run_group.create_array(
        "feret_roundness",
        data=feret_roundness,
        chunks=(min(1024, total_rois), 2),
        overwrite=True,
    )
    run_group.create_array(
        "eye_separation",
        data=eye_separation,
        chunks=(min(1024, total_rois),),
        overwrite=True,
    )
    run_group.create_array("contour_left_ptr", data=left_ptr, overwrite=True)
    run_group.create_array("contour_left_len", data=left_len, overwrite=True)
    run_group.create_array("contour_right_ptr", data=right_ptr, overwrite=True)
    run_group.create_array("contour_right_len", data=right_len, overwrite=True)
    run_group.create_array(
        "contours_left",
        data=left_store,
        chunks=(max(1, min(4096, left_store.shape[0])), 2),
        overwrite=True,
    )
    run_group.create_array(
        "contours_right",
        data=right_store,
        chunks=(max(1, min(4096, right_store.shape[0])), 2),
        overwrite=True,
    )

    git_info = get_git_info()
    env_info = get_environment_info()
    total_successful_eyes = int(ellipse_success.sum())
    pair_rate = float(successful_pairs / total_rois) if total_rois > 0 else float("nan")

    run_group.attrs.update(
        {
            "method": "yolo_eye_segmentation",
            "model_path": str(model_path_resolved),
            "model_device": model_device,
            "ultralytics_version": ultralytics_version,
            "config": {
                "batch_size": batch_size,
                "imgsz": imgsz,
                "conf": conf,
                "iou": iou,
                "max_det": max_det,
                "mask_threshold": mask_threshold,
                "use_retina_masks": use_retina_masks,
                "retina_masks_supported": retina_flag_supported,
                "prototype_fallback_used": prototype_fallback_used,
                "native_mask_hook_used": native_cache_used,
                "proto_upsample_factor": proto_factor,
            },
            "source_crop_run": crop_run_name,
            "total_rois": total_rois,
            "successful_eyes": total_successful_eyes,
            "successful_roi_pairs": int(successful_pairs),
            "successful_roi_pair_rate": pair_rate,
            "eye_labels": ["eye_left", "eye_right"],
            "git_commit": git_info.get("commit_hash", "unknown"),
            "git_branch": git_info.get("branch", "unknown"),
            "hostname": env_info["platform"].get("hostname", "unknown"),
        }
    )
    run_group.attrs["rejected_overlap"] = 0
    run_group.attrs["rejected_too_close"] = 0
    run_group.attrs["rejected_too_far"] = 0

    console.print(
        f"[green]✓[/green] Eye masks saved as [cyan]eye_masks_runs/{resolved_run_name}[/cyan] "
        f"({total_successful_eyes} successful eyes, {successful_pairs}/{total_rois} ROI pairs)"
    )

    _PROTO_UPSAMPLE_FACTOR = 1

    return resolved_run_name


__all__ = ["segment_eye_masks_yolo"]
