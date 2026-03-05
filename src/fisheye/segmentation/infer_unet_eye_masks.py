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
from ..shared.provenance_attrs import build_source_keypoints_attrs, resolve_source_keypoints_run
from ..shared.row_alignment import assert_row_alignment
from ..shared.stage_provenance import build_stage_provenance, write_stage_provenance
from ..shared.zarr.schema import add_processing_run
from ..utils.system import get_environment_info, get_git_info
from .unet import UNetSmall


def _compression_kwargs(array: zarr.Array) -> Dict[str, object]:
    """Return compression kwargs compatible with both Zarr v2 and v3."""
    kwargs: Dict[str, object] = {}

    compressors = getattr(array, "compressors", None)
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
            (f"crop_runs/{crop_run}/roi_images", crop_group["roi_images"]),
            (f"crop_runs/{crop_run}/frame_indices", crop_group.get("frame_indices")),
            (f"crop_runs/{crop_run}/detection_indices", crop_group.get("detection_indices")),
            (f"crop_runs/{crop_run}/detection_source", crop_group.get("detection_source")),
        ),
        stage="infer_unet_eye_masks input",
    )


_EYE_MASKS_STATUS_SOURCE = "runtime_infer_unet_eye_masks"


def _status_text(value: object) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", "ignore")
    text = str(value).strip()
    return text or None


def _status_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_zarr_mtime_ns(path: Path) -> Optional[int]:
    try:
        return int(path.stat().st_mtime_ns)
    except OSError:
        return None


def _clean_details(details: Dict[str, object]) -> Dict[str, object]:
    return {key: value for key, value in details.items() if value is not None}


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

    try:
        dataset_id, session_uuid = resolve_dataset_id(root, zarr_path)
        recording_id = _status_text(root.attrs.get("recording_id")) or _status_text(session_uuid)
        zarr_use = _status_text(root.attrs.get("zarr_use"))
        zarr_purpose = _status_text(root.attrs.get("zarr_purpose"))
        registry.upsert_dataset(
            dataset_id,
            session_uuid=session_uuid,
            zarr_path=zarr_path,
            recording_id=recording_id,
            zarr_use=zarr_use,
            zarr_purpose=zarr_purpose,
        )

        run_attrs: Dict[str, object] = {}
        eye_parent = root.get("eye_masks_runs")
        if run_name and eye_parent is not None and run_name in eye_parent:
            run_attrs = dict(getattr(eye_parent[run_name], "attrs", {}))

        method = _status_text(run_attrs.get("method")) or _status_text(method_hint)
        total_rois = _status_float(run_attrs.get("total_rois"))
        coverage_pct = 100.0 if total_rois is not None and total_rois > 0 and status == "ok" else None

        review_status_raw = run_attrs.get("eye_mask_review_status")
        review_status = review_status_raw if isinstance(review_status_raw, dict) else None
        details = _clean_details(
            {
                "reason": reason,
                "source_crop_run": _status_text(run_attrs.get("source_crop_run")),
                "source_eye_masks_run": _status_text(run_attrs.get("source_eye_masks_run")),
                "source_keypoints_run": _status_text(
                    run_attrs.get("source_keypoints_run") or run_attrs.get("source_keypoint_run")
                ),
                "probabilities_channels": run_attrs.get("probabilities_channels"),
                "write_binary_masks": run_attrs.get("masks_from") is not None,
                "total_rois": run_attrs.get("total_rois"),
                "inference_duration_seconds": run_attrs.get("inference_duration_seconds"),
                "requested_crop_run": _status_text(requested_crop_run),
                "error": _status_text(error_text),
            }
        )
        if isinstance(status_details, dict):
            details.update(_clean_details(dict(status_details)))

        upsert_recording_step_status(
            registry,
            dataset_id=dataset_id,
            recording_id=recording_id,
            step_name="eye_masks",
            status=status,
            run_name=run_name,
            method=method,
            coverage_pct=coverage_pct,
            review_status_json=review_status,
            details_json=details,
            source=_EYE_MASKS_STATUS_SOURCE,
            zarr_mtime_ns=_safe_zarr_mtime_ns(zarr_path),
        )
        if status == "ok":
            invalidate_downstream_steps(
                registry,
                dataset_id=dataset_id,
                step_name="eye_masks",
                source=_EYE_MASKS_STATUS_SOURCE,
                recording_id=recording_id,
                trigger_run_name=run_name,
            )
    except Exception as exc:
        if console is not None:
            console.print(
                f"[yellow]Warning:[/yellow] failed to write recording step status for eye_masks: {exc}"
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
    roi_array: zarr.Array,
    batch_size: int,
    device: torch.device,
    label_mode: str,
    console: Console,
    write_binary: bool,
) -> Tuple[int, bool]:
    total_rois = int(roi_array.shape[0])
    height = int(roi_array.shape[-2])
    width = int(roi_array.shape[-1])
    prob_channels = 1 if label_mode == "union" else 2

    chunk_size = getattr(roi_array, "chunks", None)
    chunk_rois = (
        max(1, min(512, chunk_size[0]))
        if (chunk_size and len(chunk_size) == roi_array.ndim)
        else min(512, max(1, total_rois))
    )
    compression_kwargs = _compression_kwargs(roi_array)

    expected_shape = (total_rois, prob_channels, height, width)

    mask_probs = run_group.create_array(
        "mask_probs_roi",
        shape=expected_shape,
        chunks=(chunk_rois, prob_channels, height, width),
        dtype="float16",
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

    with progress, torch.no_grad():
        for start in range(0, total_rois, batch_size):
            stop = min(start + batch_size, total_rois)
            roi_np = np.asarray(roi_array[start:stop])
            roi_np = _normalise_roi_batch(roi_np)
            imgs = torch.from_numpy(roi_np).to(device, non_blocking=True)
            if device.type == "cuda":
                imgs = imgs.contiguous(memory_format=torch.channels_last)

            amp_module = getattr(torch, "amp", None)
            if device.type == "cuda" and amp_module is not None and hasattr(amp_module, "autocast"):
                autocast_cm = amp_module.autocast("cuda")
            elif device.type == "cuda" and hasattr(torch.cuda, "amp"):
                autocast_cm = torch.cuda.amp.autocast()
            else:
                autocast_cm = nullcontext()

            with autocast_cm:
                logits = model(imgs)

            probs = torch.sigmoid(logits).float().cpu().numpy()

            if probs.ndim == 3:
                probs = probs[:, None, :, :]

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

            probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
            probs = np.clip(probs, 0.0, 1.0).astype(np.float16, copy=False)

            mask_probs[start:stop] = probs
            if masks_roi is not None:
                masks_roi[start:stop] = (probs >= 0.5).astype(np.uint8, copy=False)
            progress.advance(task, stop - start)
    return prob_channels, masks_roi is not None


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Infer eye-mask probabilities using a trained U-Net segmenter."
    )
    parser.add_argument("zarr_path", help="Path to Palette Zarr archive.")
    parser.add_argument("checkpoint", help="Path to trained U-Net checkpoint (.pt).")
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
        default=64,
        help="Batch size used during inference (default: 64).",
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

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
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

    crop_parent = root.get("crop_runs")
    crop_run = args.crop_run
    crop_resolved_from = "cli_arg" if crop_run else "unset"
    available_crop_runs = _sorted_group_keys(crop_parent)

    if src_run is not None:
        crop_run = crop_run or src_run.attrs.get("source_crop_run")
        if crop_run:
            crop_resolved_from = "source_run_attr"
    elif args.use_crop:
        if crop_parent is not None:
            crop_run = crop_run or crop_parent.attrs.get("latest")
            if crop_run:
                crop_resolved_from = "crop_latest_attr"
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
    if not crop_run and available_crop_runs:
        crop_run = available_crop_runs[-1]
        crop_resolved_from = "crop_group_latest"
    if not crop_run or crop_parent is None or crop_run not in crop_parent:
        raise ValueError(
            "Unable to resolve crop run for ROI images (use --crop-run or provide --eye-mask-run). "
            f"Available crop runs: {', '.join(available_crop_runs) if available_crop_runs else 'none'}."
        )

    crop_group = root[f"crop_runs/{crop_run}"]
    roi_path = f"crop_runs/{crop_run}/roi_images"
    if roi_path not in root:
        raise ValueError(f"ROI images array '{roi_path}' missing from Zarr store.")

    console.print(
        f"[dim]Resolved crop run: {crop_run} "
        f"(source={crop_resolved_from})[/dim]"
    )

    roi_array = root[roi_path]
    total_rois = int(roi_array.shape[0])
    if total_rois == 0:
        _emit_eye_masks_status(
            registry=registry,
            root=root,
            zarr_path=zarr_path,
            status="missing",
            reason="no_rois",
            run_name=None,
            requested_crop_run=_status_text(crop_run),
            method_hint="unet_eye_mask_segmenter",
            status_details=status_details,
            error_text=None,
            console=console,
        )
        raise ValueError("ROI image array is empty; nothing to segment.")

    _validate_input_row_alignment(
        crop_group=crop_group,
        crop_run=str(crop_run),
        total_rois=total_rois,
    )

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
        "device": str(device),
        "use_crop": bool(args.use_crop),
    }
    source_runs: Dict[str, str] = {}
    if crop_run:
        source_runs["crop"] = crop_run
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
        console.print("[dim]Copied detection_source from crop run[/dim]")
    else:
        console.print("[yellow]Crop run missing detection_source; defaulting to zeros.[/yellow]")

    start_time = time.perf_counter()
    stored_channels, wrote_binary = _write_mask_probs(
        run_group,
        model,
        roi_array,
        int(args.batch_size),
        device,
        label_mode,
        console,
        write_binary=bool(args.write_binary_masks),
    )
    duration = time.perf_counter() - start_time

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
    resolved_keypoints_run = _resolve_source_keypoints_run_for_unet(
        explicit_keypoints_run=args.keypoints_run,
        source_attrs=src_attrs,
        latest_keypoints_run=latest_keypoints,
    )

    run_group.attrs.update(src_attrs)
    run_group.attrs.update(
        {
            "method": "unet_eye_mask_segmenter",
            "segmenter": "unet",
            "segmenter_label_mode": label_mode,
            "source_eye_masks_run": source_run_name,
            "source_detect_run": crop_group.attrs.get("source_detect_run", "unknown"),
            **build_source_keypoints_attrs(resolved_keypoints_run, include_legacy_alias=True),
            "source_crop_run": crop_run,
            "detection_source_path": crop_group.attrs.get("detection_source_path"),
            "source_checkpoint": str(checkpoint_path),
            "source_checkpoint_best_val_dice": float(checkpoint.get("best_val_dice", float("nan"))),
            "total_rois": int(total_rois),
            "probabilities_dtype": "float16",
            "probabilities_channels": int(stored_channels),
            "probabilities_source": "mask_probs_roi",
            "mask_probability_threshold": 0.5,
            "inference_device": str(device),
            "inference_batch_size": int(args.batch_size),
            "inference_duration_seconds": float(duration),
            "duration_seconds": float(duration),
            "dataset_meta": dataset_meta,
            "git_commit": git_info.get("commit_hash", "unknown"),
            "git_branch": git_info.get("branch", "unknown"),
            "hostname": env_info["platform"].get("hostname", "unknown"),
        }
    )
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
        },
        inputs={
            "source_eye_masks_run": source_run_name,
            "source_crop_run": crop_run,
            "source_keypoints_run": resolved_keypoints_run,
            "frame_source": crop_group.attrs.get("video_source_type", "zarr"),
            "source_video_path": crop_group.attrs.get("video_source_path"),
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
    _emit_eye_masks_status(
        registry=registry,
        root=root,
        zarr_path=zarr_path,
        status="ok",
        reason="present",
        run_name=resolved_run_name,
        requested_crop_run=_status_text(crop_run),
        method_hint="unet_eye_mask_segmenter",
        status_details=status_details,
        error_text=None,
        console=console,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
