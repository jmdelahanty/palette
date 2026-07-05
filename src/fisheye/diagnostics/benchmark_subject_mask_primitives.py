"""Benchmark optional subject-mask primitive backends.

This diagnostic is intentionally read-only. It compares low-level mask
operations used by subject-mask finalization against optional accelerated
libraries without changing production finalizer behavior.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
import tempfile
import warnings
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.measure import label as skimage_label

from ..shared.json_safety import json_attr_safe
from ..shared.mask_geometry import fill_holes
from ..shared.mask_probability_encoding import decode_probability_values_from_attrs
from ..shared.zarr_io import open_zarr_root


_DEFAULT_COMPONENT = "subject_body"
_OPERATIONS = ("connected_components", "closing", "fill_holes")


@dataclass(frozen=True)
class PrimitiveBenchmarkResult:
    operation: str
    backend: str
    status: str
    seconds: float | None = None
    masks_per_second: float | None = None
    parity: str | None = None
    detail: str = ""


def _has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _synthetic_masks(row_count: int, height: int, width: int) -> np.ndarray:
    total = max(1, int(row_count))
    h = max(16, int(height))
    w = max(16, int(width))
    masks = np.zeros((total, h, w), dtype=np.uint8)
    for idx in range(total):
        mode = idx % 7
        cy = int((idx * 17) % max(1, h - 24)) + 12
        cx = int((idx * 23) % max(1, w - 24)) + 12
        if mode == 0:
            continue
        if mode == 1:
            masks[idx, cy - 5 : cy + 6, cx - 7 : cx + 8] = 1
        elif mode == 2:
            masks[idx, cy - 7 : cy + 8, cx - 7 : cx + 8] = 1
            masks[idx, cy - 2 : cy + 3, cx - 2 : cx + 3] = 0
        elif mode == 3:
            masks[idx, cy - 8 : cy - 2, cx - 8 : cx - 2] = 1
            masks[idx, cy + 2 : cy + 8, cx + 2 : cx + 8] = 1
        elif mode == 4:
            y0 = max(0, cy - 8)
            x0 = max(0, cx - 8)
            masks[idx, y0 : y0 + 14, x0 : x0 + 14] = np.eye(14, dtype=np.uint8)
        elif mode == 5:
            masks[idx, 0:18, cx - 8 : cx + 9] = 1
            masks[idx, 2:12, cx - 4 : cx + 5] = 0
        else:
            masks[idx, cy - 10 : cy + 11, cx - 3 : cx + 4] = 1
            masks[idx, cy - 3 : cy + 4, cx - 10 : cx + 11] = 1
    return masks


def _load_real_masks(
    zarr_path: Path,
    *,
    subject_run: str,
    component: str,
    start_row: int,
    row_count: int,
    threshold: float,
) -> np.ndarray:
    root = open_zarr_root(zarr_path, mode="r")
    group = root["subject_mask_runs"][str(subject_run)]
    labels = tuple(str(label) for label in group.attrs.get("mask_labels", ()))
    if component not in labels:
        raise KeyError(f"Component {component!r} not found in subject mask labels {labels!r}.")
    comp_idx = labels.index(component)
    total_rows = int(group["mask_probs_roi"].shape[0])
    start = max(0, int(start_row))
    stop = min(total_rows, start + max(1, int(row_count)))
    if stop <= start:
        raise ValueError(f"Requested empty row range start={start} row_count={row_count} total_rows={total_rows}.")
    values = np.asarray(group["mask_probs_roi"][start:stop, comp_idx])
    probs = decode_probability_values_from_attrs(
        values,
        attrs=group.attrs,
        source_path=f"subject_mask_runs/{subject_run}/mask_probs_roi",
    )
    return (probs >= float(threshold)).astype(np.uint8, copy=False)


def _connected_components_cv2(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = int(masks.shape[0])
    counts = np.zeros((total,), dtype=np.int32)
    largest = np.zeros((total,), dtype=np.float32)
    for idx in range(total):
        binary = (np.asarray(masks[idx], dtype=np.uint8) > 0).astype(np.uint8, copy=False)
        area = int(np.count_nonzero(binary))
        if area <= 0:
            continue
        label_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        areas = np.asarray(stats[1:, cv2.CC_STAT_AREA], dtype=np.int64)
        counts[idx] = np.int32(max(0, int(label_count) - 1))
        largest[idx] = np.float32(float(areas.max() / area) if areas.size else 0.0)
    return counts, largest


def _connected_components_cc3d(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not _has_module("cc3d"):
        raise RuntimeError("cc3d is not installed.")
    import cc3d

    total = int(masks.shape[0])
    counts = np.zeros((total,), dtype=np.int32)
    largest = np.zeros((total,), dtype=np.float32)
    for idx in range(total):
        binary = (np.asarray(masks[idx], dtype=np.uint8) > 0).astype(np.uint8, copy=False)
        area = int(np.count_nonzero(binary))
        if area <= 0:
            continue
        labels, count = cc3d.connected_components(
            binary,
            connectivity=8,
            return_N=True,
            binary_image=True,
        )
        areas = np.bincount(np.asarray(labels).reshape(-1), minlength=int(count) + 1)[1:]
        counts[idx] = np.int32(int(count))
        largest[idx] = np.float32(float(areas.max() / area) if areas.size else 0.0)
    return counts, largest


def _connected_components_skimage(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = int(masks.shape[0])
    counts = np.zeros((total,), dtype=np.int32)
    largest = np.zeros((total,), dtype=np.float32)
    for idx in range(total):
        binary = np.asarray(masks[idx], dtype=np.uint8) > 0
        area = int(np.count_nonzero(binary))
        if area <= 0:
            continue
        labels = skimage_label(binary, connectivity=2)
        count = int(labels.max())
        areas = np.bincount(np.asarray(labels).reshape(-1), minlength=count + 1)[1:]
        counts[idx] = np.int32(count)
        largest[idx] = np.float32(float(areas.max() / area) if areas.size else 0.0)
    return counts, largest


def _require_cupy_device() -> Any:
    if not _has_module("cupy"):
        raise RuntimeError("cupy is not installed.")
    import cupy as cp

    device_count = int(cp.cuda.runtime.getDeviceCount())
    if device_count <= 0:
        raise RuntimeError("No CUDA-capable device is available.")
    return cp


def _connected_components_cucim(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not _has_module("cucim"):
        raise RuntimeError("cucim is not installed.")
    cp = _require_cupy_device()
    from cucim.skimage.measure import label as cucim_label

    total = int(masks.shape[0])
    counts = np.zeros((total,), dtype=np.int32)
    largest = np.zeros((total,), dtype=np.float32)
    for idx in range(total):
        binary_np = (np.asarray(masks[idx], dtype=np.uint8) > 0).astype(np.uint8, copy=False)
        area = int(np.count_nonzero(binary_np))
        if area <= 0:
            continue
        labels = cucim_label(cp.asarray(binary_np), connectivity=2)
        count = int(cp.max(labels).get())
        areas = cp.bincount(labels.reshape(-1), minlength=count + 1)[1:]
        counts[idx] = np.int32(count)
        largest[idx] = np.float32(float(cp.max(areas).get() / area) if count > 0 else 0.0)
    return counts, largest


def _closing_cv2(masks: np.ndarray, *, radius: int) -> np.ndarray:
    if radius <= 0:
        return (np.asarray(masks, dtype=np.uint8) > 0).astype(np.uint8)
    kernel_size = int(radius) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    output = np.zeros_like(masks, dtype=np.uint8)
    for idx in range(int(masks.shape[0])):
        output[idx] = cv2.morphologyEx((masks[idx] > 0).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return output


def _closing_kornia(masks: np.ndarray, *, radius: int, device: str) -> np.ndarray:
    if not _has_module("kornia"):
        raise RuntimeError("kornia is not installed.")
    if radius <= 0:
        return (np.asarray(masks, dtype=np.uint8) > 0).astype(np.uint8)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*torch.cuda.amp.custom_fwd.*", category=FutureWarning)
        import torch
        import kornia.morphology as km

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for kornia.")
    kernel_size = int(radius) * 2 + 1
    kernel_np = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)).astype(np.float32)
    tensor = torch.as_tensor((masks > 0).astype(np.float32), device=device).unsqueeze(1)
    kernel = torch.as_tensor(kernel_np, device=device)
    closed = km.closing(tensor, kernel, border_type="geodesic", border_value=0.0)
    return (closed.squeeze(1).detach().cpu().numpy() > 0.5).astype(np.uint8)


def _closing_cucim(masks: np.ndarray, *, radius: int) -> np.ndarray:
    if not _has_module("cucim"):
        raise RuntimeError("cucim is not installed.")
    if radius <= 0:
        return (np.asarray(masks, dtype=np.uint8) > 0).astype(np.uint8)
    cp = _require_cupy_device()
    from cucim.skimage.morphology import binary_closing

    kernel_size = int(radius) * 2 + 1
    footprint = cp.asarray(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)).astype(bool))
    chunks = []
    for idx in range(int(masks.shape[0])):
        closed = binary_closing(cp.asarray(masks[idx] > 0), footprint=footprint)
        chunks.append(cp.asnumpy(closed).astype(np.uint8, copy=False))
    return np.stack(chunks, axis=0) if chunks else np.zeros_like(masks, dtype=np.uint8)


def _fill_holes_current(masks: np.ndarray) -> np.ndarray:
    output = np.zeros_like(masks, dtype=np.uint8)
    for idx in range(int(masks.shape[0])):
        output[idx] = fill_holes(masks[idx]).astype(np.uint8, copy=False)
    return output


def _fill_holes_scipy(masks: np.ndarray) -> np.ndarray:
    output = np.zeros_like(masks, dtype=np.uint8)
    for idx in range(int(masks.shape[0])):
        output[idx] = np.asarray(binary_fill_holes(masks[idx] > 0), dtype=np.uint8)
    return output


def _time_call(func: Callable[[], Any], *, repeat: int) -> tuple[Any, float]:
    best_seconds = float("inf")
    best_value: Any = None
    for _ in range(max(1, int(repeat))):
        started = time.perf_counter()
        value = func()
        elapsed = float(time.perf_counter() - started)
        if elapsed < best_seconds:
            best_seconds = elapsed
            best_value = value
    return best_value, best_seconds


def _parity_arrays(value: Any, expected: Any) -> str:
    if isinstance(value, tuple) and isinstance(expected, tuple):
        if len(value) != len(expected):
            return "fail"
        return "ok" if all(np.array_equal(a, b) for a, b in zip(value, expected)) else "fail"
    return "ok" if np.array_equal(value, expected) else "fail"


def _result(
    operation: str,
    backend: str,
    *,
    status: str,
    seconds: float | None = None,
    total_masks: int,
    parity: str | None = None,
    detail: str = "",
) -> PrimitiveBenchmarkResult:
    return PrimitiveBenchmarkResult(
        operation=operation,
        backend=backend,
        status=status,
        seconds=float(seconds) if seconds is not None else None,
        masks_per_second=float(total_masks / seconds) if seconds and seconds > 0 else None,
        parity=parity,
        detail=str(detail),
    )


def _run_backend(
    operation: str,
    backend: str,
    func: Callable[[], Any],
    *,
    expected: Any,
    repeat: int,
    total_masks: int,
) -> PrimitiveBenchmarkResult:
    try:
        value, seconds = _time_call(func, repeat=repeat)
        return _result(
            operation,
            backend,
            status="ok",
            seconds=seconds,
            total_masks=total_masks,
            parity=_parity_arrays(value, expected),
        )
    except Exception as exc:
        return _result(
            operation,
            backend,
            status="skipped",
            total_masks=total_masks,
            detail=f"{type(exc).__name__}: {exc}",
        )


def _run_cucim_worker_backend(
    operation: str,
    backend: str,
    masks: np.ndarray,
    *,
    repeat: int,
    closing_radius: int,
    total_masks: int,
) -> PrimitiveBenchmarkResult:
    with tempfile.TemporaryDirectory(prefix="palette_mask_primitive_") as tmpdir:
        input_npz = Path(tmpdir) / "masks.npz"
        np.savez_compressed(input_npz, masks=np.asarray(masks, dtype=np.uint8))
        cmd = [
            sys.executable,
            "-m",
            "fisheye.diagnostics.benchmark_subject_mask_primitives",
            "--_worker-backend",
            backend,
            "--_input-npz",
            str(input_npz),
            "--repeat",
            str(int(repeat)),
            "--closing-radius",
            str(int(closing_radius)),
        ]
        completed = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.stdout.strip():
        try:
            payload = json.loads(completed.stdout)
            result = PrimitiveBenchmarkResult(**payload)
            if completed.returncode == 0:
                return result
            return PrimitiveBenchmarkResult(
                operation=result.operation,
                backend=result.backend,
                status="skipped",
                parity=result.parity,
                seconds=result.seconds,
                masks_per_second=result.masks_per_second,
                detail=f"worker exited {completed.returncode}; stderr={completed.stderr.strip()}",
            )
        except Exception:
            pass
    return _result(
        operation,
        backend,
        status="skipped",
        total_masks=total_masks,
        detail=f"worker exited {completed.returncode}; stderr={completed.stderr.strip()}",
    )


def _run_worker_backend(args: argparse.Namespace) -> int:
    masks = np.asarray(np.load(str(args._input_npz))["masks"], dtype=np.uint8)
    backend = str(args._worker_backend)
    total_masks = int(masks.shape[0])
    if backend == "cucim_label_cuda":
        expected = _connected_components_cv2(masks)
        result = _run_backend(
            "connected_components",
            backend,
            lambda: _connected_components_cucim(masks),
            expected=expected,
            repeat=int(args.repeat),
            total_masks=total_masks,
        )
    elif backend == "cucim_binary_closing_cuda":
        expected = _closing_cv2(masks, radius=int(args.closing_radius))
        result = _run_backend(
            "closing",
            backend,
            lambda: _closing_cucim(masks, radius=int(args.closing_radius)),
            expected=expected,
            repeat=int(args.repeat),
            total_masks=total_masks,
        )
    else:
        result = _result(
            "unknown",
            backend,
            status="skipped",
            total_masks=total_masks,
            detail=f"Unsupported worker backend {backend!r}.",
        )
    print(json.dumps(asdict(result), sort_keys=True), flush=True)
    # cuCIM can segfault during interpreter teardown in this environment after
    # successful work. This worker is diagnostic-only; skip teardown after the
    # result has been flushed so the parent process receives a reliable status.
    os._exit(0)


def _run_benchmarks(
    masks: np.ndarray,
    *,
    operations: Sequence[str],
    repeat: int,
    closing_radius: int,
    include_gpu: bool,
) -> list[PrimitiveBenchmarkResult]:
    results: list[PrimitiveBenchmarkResult] = []
    total_masks = int(masks.shape[0])
    selected = set(str(op) for op in operations)

    if "connected_components" in selected:
        expected, seconds = _time_call(lambda: _connected_components_cv2(masks), repeat=repeat)
        results.append(_result(
            "connected_components",
            "cv2_connectedComponentsWithStats",
            status="ok",
            seconds=seconds,
            total_masks=total_masks,
            parity="reference",
        ))
        for backend, func in (
            ("cc3d_connected_components", _connected_components_cc3d),
            ("skimage_label", _connected_components_skimage),
        ):
            results.append(_run_backend(
                "connected_components",
                backend,
                lambda func=func: func(masks),
                expected=expected,
                repeat=repeat,
                total_masks=total_masks,
            ))
        if include_gpu:
            results.append(_run_cucim_worker_backend(
                "connected_components",
                "cucim_label_cuda",
                repeat=repeat,
                closing_radius=closing_radius,
                total_masks=total_masks,
                masks=masks,
            ))

    if "closing" in selected:
        expected, seconds = _time_call(lambda: _closing_cv2(masks, radius=closing_radius), repeat=repeat)
        results.append(_result(
            "closing",
            "cv2_morphologyEx",
            status="ok",
            seconds=seconds,
            total_masks=total_masks,
            parity="reference",
        ))
        results.append(_run_backend(
            "closing",
            "kornia_cpu",
            lambda: _closing_kornia(masks, radius=closing_radius, device="cpu"),
            expected=expected,
            repeat=repeat,
            total_masks=total_masks,
        ))
        if include_gpu:
            results.append(_run_backend(
                "closing",
                "kornia_cuda",
                lambda: _closing_kornia(masks, radius=closing_radius, device="cuda"),
                expected=expected,
                repeat=repeat,
                total_masks=total_masks,
            ))
            results.append(_run_cucim_worker_backend(
                "closing",
                "cucim_binary_closing_cuda",
                repeat=repeat,
                closing_radius=closing_radius,
                total_masks=total_masks,
                masks=masks,
            ))

    if "fill_holes" in selected:
        expected, seconds = _time_call(lambda: _fill_holes_current(masks), repeat=repeat)
        results.append(_result(
            "fill_holes",
            "palette_cv2_flood_fill",
            status="ok",
            seconds=seconds,
            total_masks=total_masks,
            parity="reference",
        ))
        results.append(_run_backend(
            "fill_holes",
            "scipy_binary_fill_holes",
            lambda: _fill_holes_scipy(masks),
            expected=expected,
            repeat=repeat,
            total_masks=total_masks,
        ))

    return results


def _parse_operations(values: Sequence[str]) -> tuple[str, ...]:
    if not values or "all" in values:
        return _OPERATIONS
    normalized = tuple(str(value) for value in values)
    unknown = sorted(set(normalized) - set(_OPERATIONS))
    if unknown:
        raise ValueError(f"Unsupported operations {unknown!r}; expected {', '.join(_OPERATIONS)} or all.")
    return normalized


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", type=Path, help="Optional analysis zarr for real subject-mask rows.")
    parser.add_argument("--subject-run", help="subject_mask_runs/<run> to read when --zarr is provided.")
    parser.add_argument("--component", default=_DEFAULT_COMPONENT, help="Mask component to read from a real run.")
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--row-count", type=int, default=256)
    parser.add_argument("--height", type=int, default=512, help="Synthetic mask height.")
    parser.add_argument("--width", type=int, default=512, help="Synthetic mask width.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for real masks.")
    parser.add_argument("--closing-radius", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--operation",
        action="append",
        default=[],
        help="Operation to benchmark: connected_components, closing, fill_holes, or all. Repeatable.",
    )
    parser.add_argument("--include-gpu", action="store_true", help="Try CUDA/GPU backends when available.")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--_worker-backend", help=argparse.SUPPRESS)
    parser.add_argument("--_input-npz", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args._worker_backend:
        if args._input_npz is None:
            parser.error("--_input-npz is required with --_worker-backend")
        return _run_worker_backend(args)

    operations = _parse_operations(args.operation)
    if args.zarr is not None:
        if not args.subject_run:
            parser.error("--subject-run is required with --zarr")
        masks = _load_real_masks(
            args.zarr,
            subject_run=str(args.subject_run),
            component=str(args.component),
            start_row=int(args.start_row),
            row_count=int(args.row_count),
            threshold=float(args.threshold),
        )
        source = {
            "kind": "zarr",
            "zarr": str(args.zarr),
            "subject_run": str(args.subject_run),
            "component": str(args.component),
            "start_row": int(args.start_row),
            "row_count": int(masks.shape[0]),
        }
    else:
        masks = _synthetic_masks(int(args.row_count), int(args.height), int(args.width))
        source = {
            "kind": "synthetic",
            "row_count": int(masks.shape[0]),
            "height": int(masks.shape[1]),
            "width": int(masks.shape[2]),
        }

    results = _run_benchmarks(
        masks,
        operations=operations,
        repeat=int(args.repeat),
        closing_radius=int(args.closing_radius),
        include_gpu=bool(args.include_gpu),
    )
    payload = {
        "schema": "palette.subject_mask_primitive_benchmark_v1",
        "source": source,
        "operations": list(operations),
        "repeat": int(args.repeat),
        "closing_radius": int(args.closing_radius),
        "optional_backends": {
            "cc3d": _has_module("cc3d"),
            "cucim": _has_module("cucim"),
            "kornia": _has_module("kornia"),
            "cupy": _has_module("cupy"),
            "torch": _has_module("torch"),
        },
        "results": [asdict(result) for result in results],
    }
    safe_payload = json_attr_safe(payload)
    text = json.dumps(safe_payload, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
