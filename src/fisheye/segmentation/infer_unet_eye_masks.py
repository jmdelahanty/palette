"""Run a trained U-Net segmenter to produce eye-mask probability maps."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import zarr
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

from ..registry.db import Registry, resolve_dataset_id
from ..registry.status_ledger import upsert_recording_step_status
from ..registry.step_cascade import invalidate_downstream_steps
from ..shared.provenance_attrs import (
    build_source_crop_snapshot_attrs,
    build_source_keypoints_attrs,
    resolve_source_keypoints_run,
)
from ..shared.crop_image_source import CropImageSource
from ..shared.inference_timing import InferenceTimingProfiler
from ..shared.registry_stage_complete import emit_stage_completion
from ..shared.row_alignment import assert_row_alignment
from ..shared.stage_provenance import build_stage_provenance, write_stage_provenance
from ..shared.type_conversions import as_float, clean_mapping, normalize_attr
from ..shared.zarr.schema import add_processing_run
from ..utils.system import get_environment_info, get_git_info
from .unet import UNetSmall


def _compression_kwargs(array: zarr.Array) -> Dict[str, object]:
    """Return compression kwargs compatible with both Zarr v2 and v3."""
    kwargs: Dict[str, object] = {}

    sentinel = object()
    compressors = getattr(array, "compressors", sentinel)
    if compressors is not sentinel:
        if compressors:
            kwargs["compressors"] = compressors
    else:
        try:
            compressor = array.compressor  # Zarr v2 API
        except (TypeError, AttributeError):
            compressor = None
        if compressor is not None:
            kwargs["compressor"] = compressor

    chunk_codecs = getattr(array, "chunk_codecs", None)
    if chunk_codecs:
        kwargs.setdefault("chunk_codecs", chunk_codecs)

    filters = getattr(array, "filters", None)
    if filters:
        kwargs.setdefault("filters", filters)

    return kwargs


def _prepare_run_group(
    root: zarr.Group,
    run_name: Optional[str],
    console: Console,
    parameters: Dict[str, object],
    source_runs: Optional[Dict[str, str]],
    env_info: Dict[str, object],
    extra_attrs: Optional[Dict[str, object]] = None,
) -> Tuple[zarr.Group, str]:
    run_group = add_processing_run(
        root,
        "eye_masks",
        parameters=parameters,
        source_runs=source_runs,
        duration_seconds=None,
        extra_attrs=extra_attrs,
        env_info=env_info,
        console=console,
        run_name=run_name,
    )
    resolved_name = run_group.name.rsplit("/", 1)[-1]
    return run_group, resolved_name


def _validate_input_row_alignment(
    *,
    crop_group: zarr.Group,
    crop_run: str,
    total_rois: int,
) -> None:
    assert_row_alignment(
        total_rois,
        (
            (f"crop_runs/{crop_run}/roi_images", crop_group.get("roi_images")),
            (f"crop_runs/{crop_run}/frame_indices", crop_group.get("frame_indices")),
            (f"crop_runs/{crop_run}/detection_indices", crop_group.get("detection_indices")),
            (f"crop_runs/{crop_run}/detection_source", crop_group.get("detection_source")),
        ),
        stage="infer_unet_eye_masks input",
    )


_EYE_MASKS_STATUS_SOURCE = "runtime_infer_unet_eye_masks"


def _status_float(value: object) -> Optional[float]:
    return as_float(value)


def _clean_details(details: Dict[str, object]) -> Dict[str, object]:
    return clean_mapping(details)


def _emit_eye_masks_status(
    *,
    registry: Optional[Registry],
    root: zarr.Group,
    zarr_path: Path,
    status: str,
    reason: str,
    run_name: Optional[str],
    requested_crop_run: Optional[str],
    method_hint: Optional[str],
    status_details: Optional[Dict[str, object]],
    error_text: Optional[str],
    console: Optional[Console],
) -> None:
    if registry is None:
        return

    run_attrs: Dict[str, object] = {}
    eye_parent = root.get("eye_masks_runs")
    if run_name and eye_parent is not None and run_name in eye_parent:
        run_attrs = dict(getattr(eye_parent[run_name], "attrs", {}))

    method = normalize_attr(run_attrs.get("method")) or normalize_attr(method_hint)
    total_rois = _status_float(run_attrs.get("total_rois"))
    coverage_pct = 100.0 if total_rois is not None and total_rois > 0 and status == "ok" else None

    review_status_raw = run_attrs.get("eye_mask_review_status")
    review_status = review_status_raw if isinstance(review_status_raw, dict) else None
    details = _clean_details(
        {
            "reason": reason,
            "source_crop_run": normalize_attr(run_attrs.get("source_crop_run")),
            "source_eye_masks_run": normalize_attr(run_attrs.get("source_eye_masks_run")),
            "source_keypoints_run": normalize_attr(
                run_attrs.get("source_keypoints_run") or run_attrs.get("source_keypoint_run")
            ),
            "probabilities_channels": run_attrs.get("probabilities_channels"),
            "write_binary_masks": run_attrs.get("masks_from") is not None,
            "total_rois": run_attrs.get("total_rois"),
            "inference_duration_seconds": run_attrs.get("inference_duration_seconds"),
            "requested_crop_run": normalize_attr(requested_crop_run),
            "error": normalize_attr(error_text),
        }
    )
    if isinstance(status_details, dict):
        details.update(_clean_details(dict(status_details)))

    emit_stage_completion(
        root,
        zarr_path,
        step_name="eye_masks",
        status=status,
        source=_EYE_MASKS_STATUS_SOURCE,
        run_name=run_name,
        method=method,
        coverage_pct=coverage_pct,
        review_status_json=review_status,
        details_json=details,
        console=console,
        warning_label="eye_masks",
        registry=registry,
        auto_registry_from_env=False,
        invalidate_on_ok=True,
        trigger_run_name=run_name,
        resolve_dataset_id_fn=resolve_dataset_id,
        upsert_step_status_fn=upsert_recording_step_status,
        invalidate_steps_fn=invalidate_downstream_steps,
    )


def _resolve_source_keypoints_run_for_unet(
    *,
    explicit_keypoints_run: Optional[str],
    source_attrs: Optional[Mapping[str, object]],
    latest_keypoints_run: Optional[str],
) -> Optional[str]:
    if explicit_keypoints_run is not None:
        return explicit_keypoints_run
    if source_attrs:
        resolved_from_source = resolve_source_keypoints_run(source_attrs)
        if resolved_from_source is not None:
            return resolved_from_source
    return latest_keypoints_run


def _resolve_existing_keypoint_source_for_unet(
    root: zarr.Group,
    *,
    explicit_keypoints_run: Optional[str],
    source_attrs: Optional[Mapping[str, object]],
    latest_keypoints_run: Optional[str],
    crop_run_name: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve an existing keypoint lineage target for an eye-mask run.

    Preference order:
    1. Explicit run if it exists.
    2. Source attrs if they reference an existing run/group.
    3. Latest raw keypoint run if it exists.
    4. Most recent raw/refined keypoint run matching the same crop run.
    """

    refined = root.get("refined_keypoints_runs")
    raw = root.get("keypoints_runs")

    def _run_exists(parent: Optional[zarr.Group], run_name: Optional[str]) -> bool:
        return parent is not None and run_name is not None and str(run_name) in parent

    def _crop_matches(parent: Optional[zarr.Group], run_name: str) -> bool:
        if parent is None or run_name not in parent or crop_run_name is None:
            return False
        attrs = getattr(parent[run_name], "attrs", {})
        return normalize_attr(attrs.get("source_crop_run")) == normalize_attr(crop_run_name)

    if explicit_keypoints_run:
        run_name = str(explicit_keypoints_run)
        if _run_exists(refined, run_name):
            return run_name, "refined_keypoints_runs"
        if _run_exists(raw, run_name):
            return run_name, "keypoints_runs"

    source_group = normalize_attr(source_attrs.get("source_keypoint_group")) if source_attrs else None
    source_run = normalize_attr(resolve_source_keypoints_run(source_attrs)) if source_attrs else None
    if source_group and source_run:
        parent = root.get(str(source_group))
        if _run_exists(parent, str(source_run)):
            return str(source_run), str(source_group)
    if source_run:
        if _run_exists(refined, str(source_run)):
            return str(source_run), "refined_keypoints_runs"
        if _run_exists(raw, str(source_run)):
            return str(source_run), "keypoints_runs"

    if latest_keypoints_run and _run_exists(raw, str(latest_keypoints_run)):
        return str(latest_keypoints_run), "keypoints_runs"

    if crop_run_name is not None:
        raw_keys = _sorted_group_keys(raw)
        for run_name in reversed(raw_keys):
            if _crop_matches(raw, run_name):
                return run_name, "keypoints_runs"
        refined_keys = _sorted_group_keys(refined)
        for run_name in reversed(refined_keys):
            if _crop_matches(refined, run_name):
                return run_name, "refined_keypoints_runs"

    return None, None


def _resolve_device(device_str: Optional[str]) -> torch.device:
    if device_str is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = str(device_str)
    if device_str.lower() == "cpu":
        return torch.device("cpu")
    if device_str.isdigit():
        return torch.device(f"cuda:{device_str}")
    return torch.device(device_str)


def _normalise_roi_batch(batch: np.ndarray) -> np.ndarray:
    if batch.ndim == 3:
        batch = batch[:, None, :, :]
    if batch.ndim != 4:
        raise ValueError(f"Unexpected ROI batch shape {batch.shape}")
    if batch.shape[1] not in (1, 3):
        raise ValueError(f"Unsupported ROI channel count: {batch.shape[1]}")
    if np.issubdtype(batch.dtype, np.integer):
        info = np.iinfo(batch.dtype)
        max_value = float(info.max)
        batch = batch.astype(np.float32, copy=False)
        if max_value > 0:
            batch /= max_value
    else:
        batch = batch.astype(np.float32, copy=False)
        max_val = float(np.nanmax(batch)) if batch.size else 0.0
        if max_val > 1.0:
            batch /= max_val
    batch = np.nan_to_num(batch, nan=0.0, posinf=1.0, neginf=0.0)
    batch = np.clip(batch, 0.0, 1.0)
    return batch


def _normalise_roi_tensor(batch: torch.Tensor) -> torch.Tensor:
    if batch.ndim == 3:
        batch = batch.unsqueeze(1)
    if batch.ndim != 4:
        raise ValueError(f"Unexpected ROI batch shape {tuple(batch.shape)}")
    if batch.shape[1] not in (1, 3):
        raise ValueError(f"Unsupported ROI channel count: {int(batch.shape[1])}")

    if not batch.is_floating_point():
        max_value = float(torch.iinfo(batch.dtype).max)
        batch = batch.to(dtype=torch.float32)
        if max_value > 0:
            batch = batch / max_value
    else:
        batch = batch.to(dtype=torch.float32)
        if batch.numel():
            max_val = torch.nan_to_num(batch, nan=0.0, posinf=float("inf"), neginf=float("-inf")).amax()
            batch = batch / torch.maximum(max_val, torch.tensor(1.0, device=batch.device, dtype=torch.float32))

    batch = torch.nan_to_num(batch, nan=0.0, posinf=1.0, neginf=0.0)
    batch = torch.clamp(batch, 0.0, 1.0)
    if batch.device.type == "cuda":
        batch = batch.contiguous(memory_format=torch.channels_last)
    return batch


def _probabilities_from_logits(
    logits: torch.Tensor,
    *,
    mask_probs_dtype: str = "float16",
) -> np.ndarray:
    probs = torch.sigmoid(logits)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
    probs = torch.clamp(probs, 0.0, 1.0)
    if mask_probs_dtype == "uint8":
        probs = torch.round(probs * 255.0).to(dtype=torch.uint8)
    else:
        probs = probs.to(dtype=torch.float16)
    return probs.cpu().numpy()


def _serialize_probabilities(probs: np.ndarray, *, mask_probs_dtype: str) -> np.ndarray:
    if mask_probs_dtype == "uint8":
        probs_float = probs.astype(np.float32, copy=False)
        return np.rint(probs_float * 255.0).clip(0.0, 255.0).astype(np.uint8)
    return probs.astype(np.float16, copy=False)


def _json_ready_meta(meta: List[Dict[str, object]]) -> List[Dict[str, object]]:
    result: List[Dict[str, object]] = []
    for item in meta:
        converted: Dict[str, object] = {}
        for key, value in item.items():
            if isinstance(value, (np.integer, np.floating)):
                converted[key] = value.item()
            elif isinstance(value, tuple):
                converted[key] = list(value)
            else:
                converted[key] = value
        result.append(converted)
    return result


def _load_checkpoint(path: Path, device: torch.device) -> Tuple[UNetSmall, Dict[str, object]]:
    checkpoint = torch.load(path, map_location=device)
    model_cfg = checkpoint.get("model_config")
    if not model_cfg:
        raise ValueError("Checkpoint missing 'model_config'; retrain with updated trainer.")
    model = UNetSmall(**model_cfg)
    state_dict = checkpoint.get("model_state")
    if state_dict is None:
        raise ValueError("Checkpoint missing 'model_state'.")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, checkpoint

def _write_mask_probs(
    run_group: zarr.Group,
    model: UNetSmall,
    roi_source: CropImageSource,
    batch_size: int,
    device: torch.device,
    label_mode: str,
    console: Console,
    write_binary: bool,
    mask_probs_chunk_rois: Optional[int] = None,
    mask_probs_dtype: str = "uint8",
    timing_profiler: Optional[InferenceTimingProfiler] = None,
) -> Tuple[int, bool]:
    total_rois = int(roi_source.total_rois)
    height, width = map(int, roi_source.roi_shape)
    prob_channels = 1 if label_mode == "union" else 2
    default_chunk_rois = min(32, max(1, total_rois))

    roi_array = roi_source.roi_array
    chunk_size = getattr(roi_array, "chunks", None)
    chunk_rois = default_chunk_rois
    if mask_probs_chunk_rois is not None:
        chunk_rois = max(1, min(int(mask_probs_chunk_rois), max(1, total_rois)))
    compression_kwargs = _compression_kwargs(roi_array) if roi_array is not None else {}
    stored_prob_dtype = np.dtype(np.uint8 if mask_probs_dtype == "uint8" else np.float16)

    expected_shape = (total_rois, prob_channels, height, width)

    mask_probs = run_group.create_array(
        "mask_probs_roi",
        shape=expected_shape,
        chunks=(chunk_rois, prob_channels, height, width),
        dtype=stored_prob_dtype,
        **compression_kwargs,
        overwrite=True,
    )

    masks_roi: Optional[zarr.Array] = None
    if write_binary:
        if "masks_roi" in run_group:
            existing = run_group["masks_roi"]
            if existing.shape == expected_shape and existing.dtype == np.uint8:
                masks_roi = existing
            else:
                del run_group["masks_roi"]
        if masks_roi is None:
            masks_roi = run_group.create_array(
                "masks_roi",
                shape=expected_shape,
                chunks=(chunk_rois, prob_channels, height, width),
                dtype="uint8",
                **compression_kwargs,
                overwrite=True,
            )

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console,
    )
    task = progress.add_task("[cyan]Running inference[/cyan]", total=total_rois)

    def _sync_cuda(stage: str, *, items: int) -> None:
        if profiler.enabled and device.type == "cuda":
            with profiler.time(stage, items=items):
                torch.cuda.synchronize(device)

    with progress, torch.no_grad():
        for start in range(0, total_rois, batch_size):
            stop = min(start + batch_size, total_rois)
            batch_count = stop - start
            profiler = timing_profiler or InferenceTimingProfiler(enabled=False)

            with profiler.time("roi_read", items=batch_count):
                roi_np = roi_source.read_slice(start, stop)

            _sync_cuda("sync_before_h2d", items=batch_count)
            with profiler.time("h2d_copy", items=batch_count):
                imgs = torch.from_numpy(roi_np).to(device, non_blocking=True)
            _sync_cuda("sync_after_h2d", items=batch_count)
            with profiler.time("input_normalize", items=batch_count):
                imgs = _normalise_roi_tensor(imgs)
            _sync_cuda("sync_after_normalize", items=batch_count)

            amp_module = getattr(torch, "amp", None)
            if device.type == "cuda" and amp_module is not None and hasattr(amp_module, "autocast"):
                autocast_cm = amp_module.autocast("cuda")
            elif device.type == "cuda" and hasattr(torch.cuda, "amp"):
                autocast_cm = torch.cuda.amp.autocast()
            else:
                autocast_cm = nullcontext()

            _sync_cuda("sync_before_forward", items=batch_count)
            with profiler.time("model_forward", items=batch_count):
                with autocast_cm:
                    logits = model(imgs)
            _sync_cuda("sync_after_forward", items=batch_count)

            _sync_cuda("sync_before_d2h", items=batch_count)
            with profiler.time("d2h_copy", items=batch_count):
                probs = _probabilities_from_logits(logits, mask_probs_dtype=mask_probs_dtype)
            _sync_cuda("sync_after_d2h", items=batch_count)

            if probs.ndim == 3:
                probs = probs[:, None, :, :]

            with profiler.time("output_postprocess", items=batch_count):
                channels = probs.shape[1]
                if label_mode == "union":
                    if channels == 1:
                        pass
                    elif channels == 2:
                        probs = np.max(probs, axis=1, keepdims=True)
                    else:
                        raise ValueError(
                            f"Union model produced {channels} channels; expected 1 or 2."
                        )
                else:  # label_mode == "lr"
                    if channels == 1:
                        raise ValueError(
                            "LR model produced a single probability channel; retrain or specify --label-mode=union."
                        )
                    elif channels != 2:
                        raise ValueError(
                            f"LR model produced {channels} channels; expected 2."
                        )

                if mask_probs_dtype == "uint8":
                    if probs.dtype != np.uint8:
                        probs_out = _serialize_probabilities(probs, mask_probs_dtype=mask_probs_dtype)
                    else:
                        probs_out = probs
                else:
                    probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float16, copy=False)
                    probs_out = probs

            with profiler.time("output_write_probs", items=batch_count):
                mask_probs[start:stop] = probs_out
            if masks_roi is not None:
                with profiler.time("output_write_binary", items=batch_count):
                    if mask_probs_dtype == "uint8":
                        masks_roi[start:stop] = (probs_out >= 128).astype(np.uint8, copy=False)
                    else:
                        masks_roi[start:stop] = (probs_out >= 0.5).astype(np.uint8, copy=False)
            with profiler.time("progress_update", items=batch_count):
                progress.advance(task, stop - start)
    return prob_channels, masks_roi is not None


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Infer eye-mask probabilities using a trained U-Net segmenter."
    )
    parser.add_argument("zarr_path", help="Path to Palette Zarr archive.")
    parser.add_argument("checkpoint", nargs="?", help="Path to trained U-Net checkpoint (.pt).")
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint_option",
        help="Path to trained U-Net checkpoint (.pt).",
    )
    parser.add_argument(
        "--eye-mask-run",
        help="Existing eye mask run to clone metadata from (optional).",
    )
    parser.add_argument(
        "--use-crop",
        action="store_true",
        default=True,
        help="When no source eye mask run is provided, prefer the latest crop run (default: true).",
    )
    parser.add_argument(
        "--crop-run",
        help="Explicit crop run providing ROI images (overrides --use-crop).",
    )
    parser.add_argument(
        "--keypoints-run",
        help="Keypoint run that produced ROI geometry (default: inferred).",
    )
    parser.add_argument(
        "--run-name",
        help="Optional name for the output run (defaults to timestamped name).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size used during inference (default: 256).",
    )
    parser.add_argument(
        "--device",
        help="Torch device to use (e.g. 'cuda:0', 'cpu'). Defaults to checkpoint device or auto.",
    )
    parser.add_argument(
        "--label-mode",
        choices=["union", "lr"],
        help="Override checkpoint label_mode (default: value stored in checkpoint).",
    )
    parser.add_argument(
        "--write-binary-masks",
        action="store_true",
        help="Also threshold and store uint8 masks alongside probabilities.",
    )
    parser.add_argument(
        "--roi-cache-policy",
        choices=("never", "auto", "always"),
        default="auto",
        help="Temporary ROI cache policy for geometry-only crop runs (default: auto).",
    )
    parser.add_argument(
        "--roi-cache-dir",
        type=Path,
        default=None,
        help="Optional scratch directory for temporary ROI caches.",
    )
    parser.add_argument(
        "--roi-live-acceleration",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Live ROI read acceleration for geometry-only crop runs (default: auto).",
    )
    parser.add_argument(
        "--roi-live-gpu-chunk-frames",
        type=int,
        default=32,
        help="Frame batch size for GPU-accelerated live ROI reads (default: 32).",
    )
    parser.add_argument(
        "--mask-probs-chunk-rois",
        type=int,
        default=32,
        help="ROI chunk length override for mask_probs_roi and masks_roi outputs (default: 32).",
    )
    parser.add_argument(
        "--mask-probs-dtype",
        choices=("float16", "uint8"),
        default="uint8",
        help="Storage dtype for mask_probs_roi (default: uint8 for analysis runs).",
    )
    parser.add_argument(
        "--profile-timings",
        action="store_true",
        help="Collect per-stage timing diagnostics and store them in the output run attrs.",
    )
    return parser


def _sorted_group_keys(group: Optional[zarr.Group]) -> List[str]:
    """Return sorted child group keys, handling Zarr API differences."""
    if group is None:
        return []
    keys_fn = getattr(group, "group_keys", None)
    try:
        keys = list(keys_fn()) if callable(keys_fn) else []
    except Exception:
        keys = []
    # Filter to strings to avoid edge cases with older stores returning bytes.
    return sorted(key for key in keys if isinstance(key, str))


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    registry: Optional[Registry] = None,
    status_details: Optional[Dict[str, object]] = None,
) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    console = Console()
    console.print("[bold cyan]Running U-Net eye-mask inference[/bold cyan]\n")

    checkpoint_value = args.checkpoint_option or args.checkpoint
    if not checkpoint_value:
        raise ValueError("U-Net eye-mask inference requires a checkpoint path.")
    checkpoint_path = Path(checkpoint_value).expanduser().resolve()
    device = _resolve_device(args.device)
    model, checkpoint = _load_checkpoint(checkpoint_path, device)
    label_mode = args.label_mode or checkpoint.get("label_mode", "union")

    zarr_path = Path(args.zarr_path).expanduser().resolve()
    root = zarr.open(str(zarr_path), mode="a")

    eye_parent = root.require_group("eye_masks_runs")
    source_run_name = args.eye_mask_run
    src_run = None
    if source_run_name:
        if source_run_name not in eye_parent:
            raise ValueError(f"Source eye mask run '{source_run_name}' not found.")
        src_run = eye_parent[source_run_name]
    else:
        latest = eye_parent.attrs.get("latest")
        if latest and latest in eye_parent and not args.crop_run:
            source_run_name = latest
            src_run = eye_parent[latest]

    crop_run = args.crop_run
    crop_resolved_from = "cli_arg" if crop_run else "unset"
    crop_parent = root.get("crop_runs")
    available_crop_runs = _sorted_group_keys(crop_parent)

    if src_run is not None:
        crop_run = crop_run or src_run.attrs.get("source_crop_run")
        if crop_run:
            crop_resolved_from = "source_run_attr"
    if (
        crop_run
        and crop_parent is not None
        and crop_run not in crop_parent
        and crop_resolved_from == "cli_arg"
    ):
        raise ValueError(
            f"Crop run '{crop_run}' not found. "
            f"Available crop runs: {', '.join(available_crop_runs) if available_crop_runs else 'none'}."
        )
    if (
        crop_run
        and crop_parent is not None
        and crop_run not in crop_parent
        and crop_resolved_from != "cli_arg"
    ):
        crop_run = None

    crop_source = CropImageSource.open(
        root,
        crop_run=str(crop_run) if crop_run is not None else None,
        zarr_path=zarr_path,
        roi_cache_policy=args.roi_cache_policy,
        roi_live_acceleration=args.roi_live_acceleration,
        roi_live_gpu_chunk_frames=args.roi_live_gpu_chunk_frames,
        roi_cache_dir=args.roi_cache_dir,
        console=console,
    )
    crop_group = crop_source.crop_group
    crop_run_name = crop_source.crop_run_name
    if crop_resolved_from == "unset":
        crop_resolved_from = "crop_source_resolver"

    console.print(
        f"[dim]Resolved crop run: {crop_run_name} "
        f"(source={crop_resolved_from})[/dim]"
    )

    total_rois = int(crop_source.total_rois)
    if total_rois == 0:
        try:
            _emit_eye_masks_status(
                registry=registry,
                root=root,
                zarr_path=zarr_path,
                status="missing",
                reason="no_rois",
                run_name=None,
                requested_crop_run=normalize_attr(crop_run_name),
                method_hint="unet_eye_mask_segmenter",
                status_details=status_details,
                error_text=None,
                console=console,
            )
        finally:
            crop_source.close()
        raise ValueError("ROI image array is empty; nothing to segment.")

    env_info = get_environment_info(
        include_all_packages=False,
        disk_path=str(zarr_path),
        collect_ip=False,
        capture_env_vars=False,
    )
    platform_info = env_info.get("platform", {})
    parameters: Dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "batch_size": int(args.batch_size),
        "label_mode": label_mode,
        "write_binary_masks": bool(args.write_binary_masks),
        "mask_probs_chunk_rois": int(args.mask_probs_chunk_rois),
        "mask_probs_dtype": str(args.mask_probs_dtype),
        "device": str(device),
        "use_crop": bool(args.use_crop),
        "profile_timings": bool(args.profile_timings),
        "roi_cache_policy": str(args.roi_cache_policy),
        "roi_live_acceleration": str(args.roi_live_acceleration),
        "roi_live_gpu_chunk_frames": int(args.roi_live_gpu_chunk_frames),
        "roi_cache_dir": str(args.roi_cache_dir) if args.roi_cache_dir else None,
    }
    source_runs: Dict[str, str] = {}
    if crop_run_name:
        source_runs["crop"] = crop_run_name
    if source_run_name:
        source_runs["eye_masks"] = source_run_name

    run_group, resolved_run_name = _prepare_run_group(
        root,
        args.run_name,
        console,
        parameters=parameters,
        source_runs=source_runs or None,
        env_info=env_info,
        extra_attrs={
            "segmenter": "unet",
            "segmenter_label_mode": label_mode,
        },
    )
    timing_profiler = InferenceTimingProfiler(enabled=bool(args.profile_timings))
    try:
        _validate_input_row_alignment(
            crop_group=crop_group,
            crop_run=str(crop_run_name),
            total_rois=total_rois,
        )

        def _copy_metadata_array(array_name: str) -> None:
            if array_name not in crop_group:
                console.print(f"[yellow]Crop run missing '{array_name}'; new eye-mask run will skip it.[/yellow]")
                return
            src = crop_group[array_name]
            data = src[:]
            chunks = getattr(src, "chunks", None)
            if not chunks:
                chunks = tuple(max(1, min(dim, 1024)) for dim in data.shape)
            else:
                chunk_list = []
                for axis, dim in enumerate(data.shape):
                    chunk_val = chunks[axis] if axis < len(chunks) else chunks[-1]
                    chunk_list.append(int(max(1, min(dim, chunk_val))))
                chunks = tuple(chunk_list)
            if array_name in run_group:
                del run_group[array_name]
            run_group.create_array(
                array_name,
                data=data,
                chunks=chunks,
                overwrite=True,
            )

        _copy_metadata_array("frame_indices")
        _copy_metadata_array("detection_indices")
        _copy_metadata_array("frame_counts")

        crop_detection_source = crop_group.get("detection_source")
        if crop_detection_source is not None and crop_detection_source.shape[0] != total_rois:
            raise ValueError(
                f"Crop run detection_source length {crop_detection_source.shape[0]} does not match ROI count {total_rois}"
            )
        detection_source = (
            crop_detection_source[:].astype(np.int8, copy=False)
            if crop_detection_source is not None
            else np.zeros(total_rois, dtype=np.int8)
        )
        run_group.create_array(
            "detection_source",
            data=detection_source,
            chunks=(min(1024, total_rois),),
            overwrite=True,
        )
        if crop_detection_source is not None:
            console.print("[dim]Copied per-ROI detection lineage (detection_source) from crop run[/dim]")
        else:
            console.print(
                "[yellow]Crop run missing per-ROI detection lineage (detection_source); defaulting to zeros.[/yellow]"
            )

        start_time = time.perf_counter()
        stored_channels, wrote_binary = _write_mask_probs(
            run_group,
            model,
            crop_source,
            int(args.batch_size),
            device,
            label_mode,
            console,
            write_binary=bool(args.write_binary_masks),
            mask_probs_chunk_rois=args.mask_probs_chunk_rois,
            mask_probs_dtype=str(args.mask_probs_dtype),
            timing_profiler=timing_profiler,
        )
        duration = time.perf_counter() - start_time
    finally:
        crop_source.close()

    git_info = get_git_info()

    dataset_meta = checkpoint.get("dataset_meta", [])
    dataset_meta = _json_ready_meta(dataset_meta) if dataset_meta else []

    src_attrs = dict(src_run.attrs) if src_run is not None else {}
    for key in list(src_attrs):
        if "refine" in key:
            src_attrs.pop(key)
    preserved_keys = {
        "parameters",
        "run_name",
        "run_stage",
        "processing_host",
        "processing_platform",
        "git_commit",
        "git_branch",
        "gpu_used",
        "gpu_device",
        "gpu_compute_capability",
        "gpu_total_memory_gb",
        "environment",
    }
    for key in list(src_attrs):
        if key in preserved_keys:
            src_attrs.pop(key)

    keypoints_parent = root.get("keypoints_runs")
    latest_keypoints = None
    if keypoints_parent is not None:
        latest_keypoints = keypoints_parent.attrs.get("latest")
    requested_keypoints_run = _resolve_source_keypoints_run_for_unet(
        explicit_keypoints_run=args.keypoints_run,
        source_attrs=src_attrs,
        latest_keypoints_run=latest_keypoints,
    )
    resolved_keypoints_run, resolved_keypoint_group = _resolve_existing_keypoint_source_for_unet(
        root,
        explicit_keypoints_run=args.keypoints_run,
        source_attrs=src_attrs,
        latest_keypoints_run=latest_keypoints,
        crop_run_name=str(crop_run_name) if crop_run_name is not None else None,
    )
    if requested_keypoints_run and resolved_keypoints_run != requested_keypoints_run:
        console.print(
            "[yellow]Requested/source keypoint lineage did not resolve to an existing run; "
            f"using {resolved_keypoint_group}/{resolved_keypoints_run} instead.[/yellow]"
            if resolved_keypoints_run and resolved_keypoint_group
            else "[yellow]Requested/source keypoint lineage did not resolve to an existing run; "
            "omitting keypoint lineage attrs.[/yellow]"
        )

    run_group.attrs.update(src_attrs)
    crop_snapshot_attrs = build_source_crop_snapshot_attrs(
        crop_group.attrs,
        source_crop_storage_mode=crop_source.storage_mode,
    )
    run_group.attrs.update(
        {
            "method": "unet_eye_mask_segmenter",
            "segmenter": "unet",
            "segmenter_label_mode": label_mode,
            "source_eye_masks_run": source_run_name,
            "source_detect_run": crop_group.attrs.get("source_detect_run", "unknown"),
            **build_source_keypoints_attrs(resolved_keypoints_run, include_legacy_alias=True),
            "source_keypoint_group": resolved_keypoint_group,
            "source_crop_run": crop_run_name,
            "detection_source_path": crop_group.attrs.get("detection_source_path"),
            **crop_snapshot_attrs,
            "source_roi_read_mode": crop_source.roi_read_mode,
            "roi_cache_policy": crop_source.roi_cache_policy,
            "source_roi_cache_used": bool(crop_source.roi_cache_used),
            "source_roi_live_acceleration_requested": crop_source.roi_live_acceleration_requested,
            "source_roi_live_acceleration_effective": crop_source.roi_live_acceleration_effective,
            "source_roi_live_acceleration_fallback_reason": crop_source.roi_live_acceleration_fallback_reason,
            "source_roi_live_gpu_chunk_frames": int(crop_source.roi_live_gpu_chunk_frames),
            "source_checkpoint": str(checkpoint_path),
            "source_checkpoint_best_val_dice": float(checkpoint.get("best_val_dice", float("nan"))),
            "total_rois": int(total_rois),
            "probabilities_dtype": str(args.mask_probs_dtype),
            "probabilities_encoding": "linear_uint8_0_255" if args.mask_probs_dtype == "uint8" else "unit_float",
            "probabilities_channels": int(stored_channels),
            "probabilities_source": "mask_probs_roi",
            "mask_probability_threshold": 0.5,
            "mask_probs_chunk_rois": int(run_group["mask_probs_roi"].chunks[0]) if "mask_probs_roi" in run_group else None,
            "inference_device": str(device),
            "inference_batch_size": int(args.batch_size),
            "inference_duration_seconds": float(duration),
            "duration_seconds": float(duration),
            "profile_timings_enabled": bool(args.profile_timings),
            "dataset_meta": dataset_meta,
            "git_commit": git_info.get("commit_hash", "unknown"),
            "git_branch": git_info.get("branch", "unknown"),
            "hostname": env_info["platform"].get("hostname", "unknown"),
        }
    )
    if timing_profiler.enabled:
        run_group.attrs["timing_profile"] = timing_profiler.summary(
            total_items=int(total_rois),
            wall_seconds=float(duration),
            notes=[
                "roi_read measures ROI slice fetch from the active crop image source.",
                "sync_before_* and sync_after_* measure explicit CUDA synchronize calls used to attribute queued GPU work deterministically.",
                "input_normalize now runs after the device transfer so dtype conversion, scaling, and clipping can execute on GPU.",
                "h2d_copy and model_forward are measured separately for the U-Net loop.",
                "d2h_copy includes sigmoid + clamp + dtype conversion (float16 or uint8) + transfer of logits back to CPU/NumPy.",
                "output_write_probs measures Zarr writes for mask probability arrays.",
                "output_write_binary measures Zarr writes for optional thresholded binary masks.",
                "progress_update measures outer progress-bar updates and related loop bookkeeping.",
            ],
        )
    if crop_source.roi_cache_key is not None:
        run_group.attrs["source_roi_cache_key"] = crop_source.roi_cache_key
    if crop_source.roi_cache_path is not None:
        run_group.attrs["source_roi_cache_path"] = crop_source.roi_cache_path
    created_timestamp = datetime.now(timezone.utc).isoformat()
    provenance_record = build_stage_provenance(
        stage="eye_masks",
        command=" ".join(sys.argv),
        created_at_utc=created_timestamp,
        version=git_info.get("short_hash") or git_info.get("commit_hash"),
        git={
            "commit": git_info.get("commit_hash"),
            "short": git_info.get("short_hash"),
            "branch": git_info.get("branch"),
            "is_dirty": git_info.get("is_dirty"),
            "remote": git_info.get("remote_url"),
        },
        environment=env_info.get("environment"),
        platform={
            "hostname": platform_info.get("hostname"),
            "system": platform_info.get("system"),
            "release": platform_info.get("release"),
            "python_version": platform_info.get("python_version"),
            "machine": platform_info.get("machine"),
        },
        parameters={
            "batch_size": int(args.batch_size),
            "device": str(device),
            "label_mode": label_mode,
            "write_binary_masks": bool(args.write_binary_masks),
            "mask_probs_chunk_rois": int(run_group.attrs.get("mask_probs_chunk_rois"))
            if run_group.attrs.get("mask_probs_chunk_rois") is not None
            else None,
            "mask_probs_dtype": str(args.mask_probs_dtype),
            "roi_cache_policy": crop_source.roi_cache_policy,
            "roi_live_acceleration": crop_source.roi_live_acceleration_requested,
            "roi_live_gpu_chunk_frames": int(crop_source.roi_live_gpu_chunk_frames),
        },
        inputs={
            "source_eye_masks_run": source_run_name,
            "source_crop_run": crop_run_name,
            "source_keypoints_run": resolved_keypoints_run,
            "frame_source": crop_source.frame_source_kind,
            "source_video_path": crop_source.frame_source_path or crop_group.attrs.get("video_source_path"),
            **crop_snapshot_attrs,
            "source_roi_read_mode": crop_source.roi_read_mode,
            "roi_live_acceleration_requested": crop_source.roi_live_acceleration_requested,
            "roi_live_acceleration_effective": crop_source.roi_live_acceleration_effective,
            "roi_live_acceleration_fallback_reason": crop_source.roi_live_acceleration_fallback_reason,
            "roi_live_gpu_chunk_frames": int(crop_source.roi_live_gpu_chunk_frames),
        },
        artifacts={
            "checkpoint_path": str(checkpoint_path),
            "segmenter": "unet",
        },
    )
    write_stage_provenance(run_group, provenance_record)

    if wrote_binary:
        run_group.attrs["masks_from"] = "threshold(mask_probs_roi, thr=0.5)"
    elif "masks_from" in run_group.attrs:
        del run_group.attrs["masks_from"]

    if not run_group.attrs.get("eye_labels"):
        run_group.attrs["eye_labels"] = ["eye_left", "eye_right"]

    output_desc = "probabilities + binary masks" if wrote_binary else "probabilities"
    console.print(
        f"\n[green]✓[/green] U-Net {output_desc} written to "
        f"[cyan]eye_masks_runs/{resolved_run_name}/mask_probs_roi[/cyan] "
        f"({total_rois:,} ROIs processed in {duration:.1f}s)."
    )
    if timing_profiler.enabled:
        console.print("[bold]Timing Profile:[/bold]")
        for line in timing_profiler.render_lines(total_items=total_rois, wall_seconds=duration, limit=6):
            console.print(f"[dim]{line}[/dim]")
    _emit_eye_masks_status(
        registry=registry,
        root=root,
        zarr_path=zarr_path,
        status="ok",
        reason="present",
        run_name=resolved_run_name,
        requested_crop_run=normalize_attr(crop_run_name),
        method_hint="unet_eye_mask_segmenter",
        status_details=status_details,
        error_text=None,
        console=console,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
