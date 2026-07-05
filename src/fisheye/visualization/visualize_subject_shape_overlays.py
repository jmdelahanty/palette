"""Render subject-shape centerline and tail-anchor overlays."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import zarr
from skimage.measure import find_contours
from skimage.morphology import skeletonize

from fisheye.analysis.subject_shape_runs import _longest_skeleton_endpoint_path_xy
from fisheye.shared.crop_image_source import CropImageSource
from fisheye.shared.mask_store import MaskStore, open_mask_store
from fisheye.shared.zarr_io import open_zarr_root


DEFAULT_OUTPUT_DIR = Path("/tmp/palette_subject_shape_overlays")
ContourSource = Literal["mask", "persisted", "auto", "compare"]
CONTOUR_SOURCE_CHOICES: tuple[str, ...] = ("mask", "persisted", "auto", "compare")
SkeletonStyle = Literal["underlay", "offset", "branches"]
SKELETON_STYLE_CHOICES: tuple[str, ...] = ("underlay", "offset", "branches")


@dataclass(frozen=True)
class SubjectShapeOverlayContext:
    root: zarr.Group
    shape_run_name: str
    shape_group: zarr.Group
    refined_run_name: str
    refined_group: zarr.Group
    label_map: dict[str, int]
    mask_store: MaskStore
    crop_source: CropImageSource | None = None


def _open_zarr(zarr_path: Path) -> zarr.Group:
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr archive not found: {zarr_path}")
    return open_zarr_root(zarr_path, mode="r")


def _resolve_shape_run(root: zarr.Group, shape_run: str | None) -> tuple[str, zarr.Group]:
    analysis = root.get("analysis")
    if analysis is None or "subject_shape_runs" not in analysis:
        raise ValueError("Archive has no analysis/subject_shape_runs group.")
    parent = analysis["subject_shape_runs"]
    run_name = str(shape_run or parent.attrs.get("latest") or "")
    if not run_name:
        candidates = sorted(str(name) for name in parent.keys())
        if not candidates:
            raise ValueError("analysis/subject_shape_runs has no runs.")
        run_name = candidates[-1]
    if run_name not in parent:
        raise ValueError(f"analysis/subject_shape_runs/{run_name} not found.")
    return run_name, parent[run_name]


def _resolve_refined_run(
    root: zarr.Group,
    shape_group: zarr.Group,
    refined_run: str | None,
) -> tuple[str, zarr.Group]:
    parent = root.get("refined_subject_masks_runs")
    if parent is None:
        raise ValueError("Archive has no refined_subject_masks_runs group.")
    run_name = str(refined_run or shape_group.attrs.get("source_refined_subject_masks_run") or parent.attrs.get("latest") or "")
    if not run_name:
        candidates = sorted(str(name) for name in parent.keys())
        if not candidates:
            raise ValueError("refined_subject_masks_runs has no runs.")
        run_name = candidates[-1]
    if run_name not in parent:
        raise ValueError(f"refined_subject_masks_runs/{run_name} not found.")
    return run_name, parent[run_name]


def _label_index_map(refined_group: zarr.Group) -> dict[str, int]:
    labels = refined_group.attrs.get("mask_labels")
    if not isinstance(labels, (list, tuple)):
        raise ValueError("Refined subject-mask run is missing mask_labels.")
    return {str(label): int(idx) for idx, label in enumerate(labels)}


def open_subject_shape_overlay_context(
    zarr_path: Path,
    *,
    shape_run: str | None = None,
    refined_run: str | None = None,
    crop_run: str | None = None,
    use_crop_images: bool = True,
) -> SubjectShapeOverlayContext:
    root = _open_zarr(zarr_path)
    shape_run_name, shape_group = _resolve_shape_run(root, shape_run)
    refined_run_name, refined_group = _resolve_refined_run(root, shape_group, refined_run)
    label_map = _label_index_map(refined_group)
    mask_store = open_mask_store(
        refined_group,
        source_path=f"refined_subject_masks_runs/{refined_run_name}",
        prefer="dense",
    )
    crop_source: CropImageSource | None = None
    if use_crop_images:
        try:
            crop_source = CropImageSource.open(root, crop_run=crop_run, zarr_path=zarr_path)
        except Exception:
            crop_source = None
    return SubjectShapeOverlayContext(
        root=root,
        shape_run_name=shape_run_name,
        shape_group=shape_group,
        refined_run_name=refined_run_name,
        refined_group=refined_group,
        label_map=label_map,
        mask_store=mask_store,
        crop_source=crop_source,
    )


def _normalize_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros(arr.shape, dtype=np.float32)
    lo = float(np.nanpercentile(arr[finite], 1.0))
    hi = float(np.nanpercentile(arr[finite], 99.0))
    if hi <= lo:
        lo = float(np.nanmin(arr[finite]))
        hi = float(np.nanmax(arr[finite]))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def _decode_reason(row: object) -> str:
    try:
        raw = bytes(np.asarray(row, dtype=np.uint8))
    except Exception:
        return ""
    return raw.split(b"\0", 1)[0].decode("utf-8", errors="replace")


def _get_array_or_none(group: zarr.Group, path: str) -> object | None:
    current: object = group
    for part in path.split("/"):
        if not isinstance(current, zarr.Group) or part not in current:
            return None
        current = current[part]
    return current


def _row_value(group: zarr.Group, path: str, row: int, default: object = None) -> object:
    array = _get_array_or_none(group, path)
    if array is None:
        return default
    try:
        return array[int(row)]
    except Exception:
        return default


def _mask_for(ctx: SubjectShapeOverlayContext, row: int, label: str) -> np.ndarray | None:
    idx = ctx.label_map.get(label)
    if idx is None:
        return None
    if int(row) < 0 or int(row) >= int(ctx.mask_store.n_rows):
        raise IndexError(f"Row {row} out of range for mask store with {ctx.mask_store.n_rows} rows.")
    mask = ctx.mask_store.read_dense(rows=int(row), channels=int(idx))[0, 0]
    return np.asarray(mask, dtype=np.uint8) > 0


def _base_image(ctx: SubjectShapeOverlayContext, row: int, subject_body: np.ndarray | None) -> tuple[np.ndarray, str]:
    if ctx.crop_source is not None and 0 <= int(row) < ctx.crop_source.total_rois:
        try:
            return _normalize_image(ctx.crop_source[int(row)]), f"crop:{ctx.crop_source.crop_run_name}"
        except Exception:
            pass
    if subject_body is not None:
        return np.asarray(subject_body, dtype=np.float32) * 0.55, "subject_body_mask"
    raise ValueError("No crop image or subject-body mask available for overlay background.")


def _plot_contours(
    ax: plt.Axes,
    mask: np.ndarray | None,
    *,
    color: str,
    linewidth: float,
    label: str,
    linestyle: str = "-",
    alpha: float = 1.0,
) -> None:
    if mask is None or not np.any(mask):
        return
    for contour in find_contours(np.asarray(mask, dtype=np.float32), level=0.5):
        if int(contour.shape[0]) < 2:
            continue
        ax.plot(
            contour[:, 1],
            contour[:, 0],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
            label=label,
        )
        label = "_nolegend_"


def _persisted_contour_for(ctx: SubjectShapeOverlayContext, row: int, component: str) -> np.ndarray | None:
    contours_group = _get_array_or_none(ctx.refined_group, f"components/{component}/contours")
    if not isinstance(contours_group, zarr.Group):
        return None
    ptr_arr = contours_group.get("ptr")
    len_arr = contours_group.get("len")
    points_arr = contours_group.get("points_xy")
    if ptr_arr is None or len_arr is None or points_arr is None:
        return None
    row = int(row)
    if row < 0 or row >= int(ptr_arr.shape[0]) or row >= int(len_arr.shape[0]):
        return None
    try:
        start = int(ptr_arr[row])
        length = int(len_arr[row])
    except Exception:
        return None
    if start < 0 or length <= 1:
        return None
    try:
        points = np.asarray(points_arr[start : start + length], dtype=np.float32).reshape(-1, 2)
    except Exception:
        return None
    if int(points.shape[0]) < 2 or not np.all(np.isfinite(points)):
        return None
    return points


def _plot_contour_points(
    ax: plt.Axes,
    points_xy: np.ndarray | None,
    *,
    color: str,
    linewidth: float,
    label: str,
    linestyle: str = "--",
    alpha: float = 1.0,
) -> None:
    if points_xy is None:
        return
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    if int(points.shape[0]) < 2:
        return
    ax.plot(
        points[:, 0],
        points[:, 1],
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        label=label,
    )


def _plot_component_contour(
    ax: plt.Axes,
    ctx: SubjectShapeOverlayContext,
    *,
    row: int,
    component: str,
    mask: np.ndarray | None,
    color: str,
    linewidth: float,
    label: str,
    contour_source: ContourSource,
) -> None:
    persisted = None
    if contour_source in {"persisted", "auto", "compare"}:
        persisted = _persisted_contour_for(ctx, row, component)

    if contour_source == "mask":
        _plot_contours(ax, mask, color=color, linewidth=linewidth, label=label)
        return

    if contour_source == "persisted":
        _plot_contour_points(
            ax,
            persisted,
            color=color,
            linewidth=linewidth,
            label=f"{label} persisted",
        )
        return

    if contour_source == "auto":
        if persisted is not None:
            _plot_contour_points(
                ax,
                persisted,
                color=color,
                linewidth=linewidth,
                label=f"{label} persisted",
            )
        else:
            _plot_contours(ax, mask, color=color, linewidth=linewidth, label=label)
        return

    if contour_source == "compare":
        _plot_contours(
            ax,
            mask,
            color=color,
            linewidth=max(0.8, linewidth * 0.8),
            label=f"{label} from mask",
            alpha=0.65,
        )
        _plot_contour_points(
            ax,
            persisted,
            color=color,
            linewidth=max(1.2, linewidth),
            label=f"{label} persisted",
            linestyle="--",
        )
        return

    raise ValueError(f"Unsupported contour_source: {contour_source!r}")


def _plot_point(
    ax: plt.Axes,
    xy: object,
    *,
    color: str,
    marker: str,
    label: str,
    size: float = 48,
) -> None:
    arr = np.asarray(xy, dtype=np.float64).reshape(-1)
    if arr.size < 2 or not np.all(np.isfinite(arr[:2])):
        return
    kwargs = {"c": color, "marker": marker, "s": size, "label": label, "linewidths": 0.5}
    if marker not in {"x", "+", "|", "_"}:
        kwargs["edgecolors"] = "black"
    ax.scatter([arr[0]], [arr[1]], **kwargs)


def _plot_centerline(ax: plt.Axes, centerline_xy: object, *, valid: bool) -> None:
    arr = np.asarray(centerline_xy, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return
    finite = np.all(np.isfinite(arr), axis=1)
    if int(np.count_nonzero(finite)) < 2:
        return
    arr = arr[finite]
    color = "#00e676" if valid else "#ff9100"
    ax.plot(arr[:, 0], arr[:, 1], color=color, linewidth=2.0, label="centerline")


def _plot_polyline(
    ax: plt.Axes,
    xy: object,
    *,
    label: str,
    color: str,
    linewidth: float = 2.0,
    linestyle: str = "-",
    alpha: float = 1.0,
) -> bool:
    arr = np.asarray(xy, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return False
    finite = np.all(np.isfinite(arr), axis=1)
    if int(np.count_nonzero(finite)) < 2:
        return False
    arr = arr[finite]
    ax.plot(
        arr[:, 0],
        arr[:, 1],
        color=color,
        linewidth=float(linewidth),
        linestyle=linestyle,
        alpha=float(alpha),
        label=label,
    )
    return True


def _scatter_points(
    ax: plt.Axes,
    xy: object,
    *,
    label: str,
    color: str,
    marker: str = "o",
    size: float = 18.0,
    alpha: float = 1.0,
) -> bool:
    arr = np.asarray(xy, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return False
    finite = np.all(np.isfinite(arr), axis=1)
    if int(np.count_nonzero(finite)) < 1:
        return False
    arr = arr[finite]
    kwargs = {
        "c": color,
        "marker": marker,
        "s": float(size),
        "label": label,
        "alpha": float(alpha),
        "linewidths": 0.5,
    }
    if marker not in {"x", "+", "|", "_"}:
        kwargs["edgecolors"] = "black"
    ax.scatter(arr[:, 0], arr[:, 1], **kwargs)
    return True


def _plot_tail_normals(
    ax: plt.Axes,
    tail_sample_xy: object,
    tail_normal_xy: object,
    *,
    length_px: float = 5.0,
) -> bool:
    points = np.asarray(tail_sample_xy, dtype=np.float64)
    normals = np.asarray(tail_normal_xy, dtype=np.float64)
    if points.ndim != 2 or normals.ndim != 2 or points.shape != normals.shape or points.shape[1] != 2:
        return False
    finite = np.all(np.isfinite(points), axis=1) & np.all(np.isfinite(normals), axis=1)
    if int(np.count_nonzero(finite)) < 1:
        return False
    points = points[finite]
    normals = normals[finite]
    half = float(length_px) / 2.0
    for idx, (point, normal) in enumerate(zip(points, normals)):
        start = point - normal * half
        stop = point + normal * half
        ax.plot(
            [start[0], stop[0]],
            [start[1], stop[1]],
            color="#ffca28",
            linewidth=0.7,
            alpha=0.85,
            label="tail normals" if idx == 0 else "_nolegend_",
        )
    return True


def _scatter_skeleton_points(
    ax: plt.Axes,
    coords_yx: np.ndarray,
    *,
    label: str,
    alpha: float,
    size: float,
    offset_xy: tuple[float, float] = (0.0, 0.0),
) -> None:
    if int(coords_yx.shape[0]) == 0:
        return
    ax.scatter(
        coords_yx[:, 1] + float(offset_xy[0]),
        coords_yx[:, 0] + float(offset_xy[1]),
        c="#ff2bd6",
        s=float(size),
        marker="s",
        linewidths=0,
        alpha=float(alpha),
        label=label,
    )


def _plot_skeleton(
    ax: plt.Axes,
    mask: np.ndarray | None,
    *,
    skeleton_style: SkeletonStyle = "underlay",
    skeleton_offset_px: float = 1.5,
) -> None:
    if mask is None or not np.any(mask):
        return
    if skeleton_style not in SKELETON_STYLE_CHOICES:
        raise ValueError(f"skeleton_style must be one of {SKELETON_STYLE_CHOICES}; got {skeleton_style!r}")
    skeleton = skeletonize(np.asarray(mask, dtype=bool))
    coords_yx = np.argwhere(skeleton)
    if int(coords_yx.shape[0]) == 0:
        return
    if skeleton_style == "underlay":
        _scatter_skeleton_points(ax, coords_yx, label="body skeleton", alpha=0.78, size=4.0)
        return
    if skeleton_style == "offset":
        _scatter_skeleton_points(
            ax,
            coords_yx,
            label="body skeleton offset",
            alpha=0.78,
            size=4.0,
            offset_xy=(float(skeleton_offset_px), float(skeleton_offset_px)),
        )
        return

    path_xy, _reason = _longest_skeleton_endpoint_path_xy(np.asarray(mask, dtype=bool))
    if path_xy is None:
        _scatter_skeleton_points(ax, coords_yx, label="body skeleton branches unavailable", alpha=0.78, size=4.0)
        return
    selected_yx = {
        (int(round(float(y))), int(round(float(x))))
        for x, y in np.asarray(path_xy, dtype=np.float64).reshape(-1, 2)
        if np.isfinite(x) and np.isfinite(y)
    }
    unused = np.asarray(
        [(int(y), int(x)) for y, x in coords_yx if (int(y), int(x)) not in selected_yx],
        dtype=np.int64,
    ).reshape(-1, 2)
    _scatter_skeleton_points(ax, coords_yx, label="body skeleton all", alpha=0.18, size=3.0)
    _scatter_skeleton_points(ax, unused, label="unused skeleton branches", alpha=0.9, size=5.0)


def _plot_body_frame(ax: plt.Axes, shape_group: zarr.Group, row: int) -> None:
    valid = bool(_row_value(shape_group, "body_frame/valid", row, False))
    if not valid:
        return
    origin = np.asarray(_row_value(shape_group, "body_frame/origin_xy", row, [np.nan, np.nan]), dtype=np.float64)
    forward = np.asarray(_row_value(shape_group, "body_frame/forward_axis_xy", row, [np.nan, np.nan]), dtype=np.float64)
    left = np.asarray(_row_value(shape_group, "body_frame/left_axis_xy", row, [np.nan, np.nan]), dtype=np.float64)
    if origin.shape != (2,) or forward.shape != (2,) or not np.all(np.isfinite(origin)) or not np.all(np.isfinite(forward)):
        return
    ax.arrow(origin[0], origin[1], forward[0] * 35, forward[1] * 35, color="#ffd54f", width=0.8, label="forward")
    if left.shape == (2,) and np.all(np.isfinite(left)):
        ax.arrow(origin[0], origin[1], left[0] * 20, left[1] * 20, color="#80cbc4", width=0.5, label="left")


def render_subject_shape_overlay(
    ctx: SubjectShapeOverlayContext,
    *,
    row: int,
    figsize: tuple[float, float] = (7.0, 7.0),
    contour_source: ContourSource = "mask",
    show_skeleton: bool = False,
    skeleton_style: SkeletonStyle = "underlay",
    skeleton_offset_px: float = 1.5,
    show_bspline: bool = False,
    show_tail_samples: bool = False,
    show_tail_normals: bool = False,
    show_spline_control_points: bool = False,
) -> plt.Figure:
    row = int(row)
    if contour_source not in CONTOUR_SOURCE_CHOICES:
        raise ValueError(f"contour_source must be one of {CONTOUR_SOURCE_CHOICES}; got {contour_source!r}")
    if skeleton_style not in SKELETON_STYLE_CHOICES:
        raise ValueError(f"skeleton_style must be one of {SKELETON_STYLE_CHOICES}; got {skeleton_style!r}")
    body = _mask_for(ctx, row, "subject_body")
    swim = _mask_for(ctx, row, "swim_bladder")
    eye_left = _mask_for(ctx, row, "eye_left")
    eye_right = _mask_for(ctx, row, "eye_right")
    base, base_source = _base_image(ctx, row, body)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(base, cmap="gray", interpolation="nearest")
    _plot_component_contour(
        ax,
        ctx,
        row=row,
        component="subject_body",
        mask=body,
        color="white",
        linewidth=1.2,
        label="body contour",
        contour_source=contour_source,
    )
    _plot_component_contour(
        ax,
        ctx,
        row=row,
        component="swim_bladder",
        mask=swim,
        color="#00bcd4",
        linewidth=1.5,
        label="swim bladder",
        contour_source=contour_source,
    )
    _plot_component_contour(
        ax,
        ctx,
        row=row,
        component="eye_left",
        mask=eye_left,
        color="#ef5350",
        linewidth=1.2,
        label="eye left",
        contour_source=contour_source,
    )
    _plot_component_contour(
        ax,
        ctx,
        row=row,
        component="eye_right",
        mask=eye_right,
        color="#42a5f5",
        linewidth=1.2,
        label="eye right",
        contour_source=contour_source,
    )

    if show_skeleton:
        _plot_skeleton(
            ax,
            body,
            skeleton_style=skeleton_style,
            skeleton_offset_px=float(skeleton_offset_px),
        )

    shape = ctx.shape_group
    centerline_valid = bool(_row_value(shape, "components/subject_body/centerline_valid", row, False))
    _plot_centerline(
        ax,
        _row_value(shape, "components/subject_body/centerline_xy", row, None),
        valid=centerline_valid,
    )
    bspline_drawn = False
    control_points_drawn = False
    tail_samples_drawn = False
    tail_normals_drawn = False
    if show_bspline:
        bspline_drawn = _plot_polyline(
            ax,
            _row_value(shape, "components/subject_body/bspline_sample_xy", row, None),
            label="B-spline sample",
            color="#ff1744",
            linewidth=1.8,
            linestyle="-",
            alpha=0.9,
        )
    if show_spline_control_points:
        control_points_drawn = _scatter_points(
            ax,
            _row_value(shape, "components/subject_body/bspline_control_points_xy", row, None),
            label="B-spline control points",
            color="#ff80ab",
            marker="D",
            size=22.0,
            alpha=0.85,
        )
    if show_tail_samples:
        tail_samples_drawn = _scatter_points(
            ax,
            _row_value(shape, "components/subject_body/tail_sample_xy", row, None),
            label="tail samples",
            color="#18ffff",
            marker="o",
            size=16.0,
            alpha=0.9,
        )
    if show_tail_normals:
        tail_normals_drawn = _plot_tail_normals(
            ax,
            _row_value(shape, "components/subject_body/tail_sample_xy", row, None),
            _row_value(shape, "components/subject_body/tail_normal_xy", row, None),
        )
    _plot_body_frame(ax, shape, row)
    snout_array = _get_array_or_none(shape, "components/subject_body/snout_tip_xy")
    snout_xy = _row_value(shape, "components/subject_body/snout_tip_xy", row, None) if snout_array is not None else None
    head_xy = _row_value(shape, "components/subject_body/head_endpoint_xy", row, None)
    if snout_array is not None:
        _plot_point(
            ax,
            snout_xy,
            color="#ff6d00",
            marker="o",
            label="snout tip",
            size=90,
        )
    _plot_point(
        ax,
        _row_value(shape, "components/swim_bladder/caudal_contour_point_xy", row, None),
        color="#ffeb3b",
        marker="*",
        label="caudal swim anchor",
        size=110,
    )
    snout_arr = np.asarray(snout_xy, dtype=np.float64) if snout_xy is not None else np.asarray([], dtype=np.float64)
    head_arr = np.asarray(head_xy, dtype=np.float64) if head_xy is not None else np.asarray([], dtype=np.float64)
    head_overlaps_snout = (
        snout_arr.shape == (2,)
        and head_arr.shape == (2,)
        and np.all(np.isfinite(snout_arr))
        and np.all(np.isfinite(head_arr))
        and float(np.linalg.norm(head_arr - snout_arr)) <= 1e-4
    )
    if not head_overlaps_snout:
        _plot_point(
            ax,
            head_xy,
            color="#00e676",
            marker="o",
            label="head endpoint",
        )
    _plot_point(
        ax,
        _row_value(shape, "components/subject_body/tail_base_xy", row, None),
        color="#ffd54f",
        marker="o",
        label="tail base",
    )
    _plot_point(
        ax,
        _row_value(shape, "components/subject_body/tail_tip_xy", row, None),
        color="#e040fb",
        marker="x",
        label="tail tip",
        size=80,
    )

    frame = _row_value(shape, "row_index/frame_indices", row, None)
    body_reason = _decode_reason(_row_value(shape, "components/subject_body/centerline_failure_reason_bytes", row, []))
    snout_reason = _decode_reason(_row_value(shape, "components/subject_body/snout_tip_failure_reason_bytes", row, []))
    snout_check_reason = _decode_reason(
        _row_value(shape, "components/subject_body/centerline_snout_check_reason_bytes", row, [])
    )
    tail_reason = _decode_reason(_row_value(shape, "components/subject_body/tail_base_failure_reason_bytes", row, []))
    bspline_reason = _decode_reason(_row_value(shape, "components/subject_body/bspline_failure_reason_bytes", row, []))
    tail_sample_reason = _decode_reason(_row_value(shape, "components/subject_body/tail_sample_failure_reason_bytes", row, []))
    anchor_reason = _decode_reason(
        _row_value(shape, "components/swim_bladder/caudal_contour_failure_reason_bytes", row, [])
    )
    body_len = _row_value(shape, "components/subject_body/body_arclength_px", row, np.nan)
    tail_len = _row_value(shape, "components/subject_body/tail_segment_arclength_px", row, np.nan)
    bspline_len = _row_value(shape, "components/subject_body/bspline_arc_length_px", row, np.nan)
    snout_distance = _row_value(shape, "components/subject_body/head_endpoint_to_snout_distance_px", row, np.nan)
    title = f"Subject shape overlay | row {row}"
    if frame is not None:
        title += f" | frame {int(frame)}"
    ax.set_title(title)
    text_lines = [
        f"shape: {ctx.shape_run_name}",
        f"refined: {ctx.refined_run_name}",
        f"background: {base_source}",
        f"contours: {contour_source}",
        f"skeleton: {bool(show_skeleton)} ({skeleton_style})",
        f"centerline: {centerline_valid} ({body_reason or 'n/a'})",
    ]
    if show_bspline or show_spline_control_points:
        text_lines.extend(
            [
                f"bspline: {bool(_row_value(shape, 'components/subject_body/bspline_valid', row, False))} "
                f"({bspline_reason or 'unavailable'})",
                f"bspline_drawn: {bspline_drawn} control_points: {control_points_drawn}",
            ]
        )
    if show_tail_samples or show_tail_normals:
        text_lines.extend(
            [
                f"tail_samples: {bool(_row_value(shape, 'components/subject_body/tail_sample_valid', row, False))} "
                f"({tail_sample_reason or 'unavailable'})",
                f"tail_samples_drawn: {tail_samples_drawn} tail_normals_drawn: {tail_normals_drawn}",
            ]
        )
    if snout_array is not None:
        text_lines.append(
            f"snout_tip: {bool(_row_value(shape, 'components/subject_body/snout_tip_valid', row, False))} "
            f"({snout_reason or 'n/a'})"
        )
    if _get_array_or_none(shape, "components/subject_body/head_endpoint_to_snout_distance_px") is not None:
        reaches = bool(_row_value(shape, "components/subject_body/centerline_reaches_snout", row, False))
        distance_text = f"{float(snout_distance):.2f}" if np.isfinite(float(snout_distance)) else "n/a"
        text_lines.append(
            f"head_to_snout_px: {distance_text} reaches_snout: {reaches} ({snout_check_reason or 'n/a'})"
        )
    text_lines.extend(
        [
            f"tail_base: {bool(_row_value(shape, 'components/subject_body/tail_base_valid', row, False))} ({tail_reason or 'n/a'})",
            f"caudal_anchor: {bool(_row_value(shape, 'components/swim_bladder/caudal_contour_valid', row, False))} ({anchor_reason or 'n/a'})",
            f"body_len_px: {float(body_len):.2f}" if np.isfinite(float(body_len)) else "body_len_px: n/a",
        ]
    )
    if show_bspline or show_spline_control_points:
        text_lines.append(
            f"bspline_len_px: {float(bspline_len):.2f}" if np.isfinite(float(bspline_len)) else "bspline_len_px: n/a"
        )
    text_lines.append(f"tail_len_px: {float(tail_len):.2f}" if np.isfinite(float(tail_len)) else "tail_len_px: n/a")
    text = "\n".join(text_lines)
    ax.text(
        0.02,
        0.02,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="white",
        bbox={"boxstyle": "round,pad=0.3", "fc": "black", "alpha": 0.55},
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique: dict[str, object] = {}
        for handle, label in zip(handles, labels):
            if label and label != "_nolegend_" and label not in unique:
                unique[label] = handle
        ax.legend(unique.values(), unique.keys(), loc="upper right", fontsize=7)
    ax.set_xlim(-0.5, base.shape[1] - 0.5)
    ax.set_ylim(base.shape[0] - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


def export_subject_shape_overlays(
    zarr_path: Path,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    shape_run: str | None = None,
    refined_run: str | None = None,
    crop_run: str | None = None,
    rows: Sequence[int] = (0,),
    use_crop_images: bool = True,
    dpi: int = 160,
    contour_source: ContourSource = "mask",
    show_skeleton: bool = False,
    skeleton_style: SkeletonStyle = "underlay",
    skeleton_offset_px: float = 1.5,
    show_bspline: bool = False,
    show_tail_samples: bool = False,
    show_tail_normals: bool = False,
    show_spline_control_points: bool = False,
) -> list[Path]:
    ctx = open_subject_shape_overlay_context(
        zarr_path,
        shape_run=shape_run,
        refined_run=refined_run,
        crop_run=crop_run,
        use_crop_images=use_crop_images,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for row in rows:
        fig = render_subject_shape_overlay(
            ctx,
            row=int(row),
            contour_source=contour_source,
            show_skeleton=show_skeleton,
            skeleton_style=skeleton_style,
            skeleton_offset_px=float(skeleton_offset_px),
            show_bspline=show_bspline,
            show_tail_samples=show_tail_samples,
            show_tail_normals=show_tail_normals,
            show_spline_control_points=show_spline_control_points,
        )
        out = output_dir / f"subject_shape_overlay_{ctx.shape_run_name}_row_{int(row):06d}.png"
        fig.savefig(out, dpi=int(dpi), bbox_inches="tight")
        plt.close(fig)
        paths.append(out)
    return paths


def _parse_rows(values: Sequence[str] | None, *, start_row: int, count: int) -> list[int]:
    if values:
        rows: list[int] = []
        for value in values:
            for part in str(value).split(","):
                part = part.strip()
                if part:
                    rows.append(int(part))
        return rows
    return list(range(int(start_row), int(start_row) + max(1, int(count))))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export subject-shape centerline/tail-anchor overlay PNGs.")
    parser.add_argument("zarr_path", type=Path, help="Palette analysis Zarr archive.")
    parser.add_argument("--shape-run", help="analysis/subject_shape_runs/<run>; defaults to latest.")
    parser.add_argument("--refined-run", help="refined_subject_masks_runs/<run>; defaults to shape source.")
    parser.add_argument("--crop-run", help="Optional crop_runs/<run> for ROI image background.")
    parser.add_argument("--no-crop-images", action="store_true", help="Use subject-body mask background only.")
    parser.add_argument("--rows", nargs="+", help="Row indices to render, e.g. --rows 0 10 20 or --rows 0,10,20.")
    parser.add_argument("--start-row", type=int, default=0, help="First row when --rows is omitted.")
    parser.add_argument("--count", type=int, default=6, help="Number of sequential rows when --rows is omitted.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument(
        "--contour-source",
        choices=CONTOUR_SOURCE_CHOICES,
        default="mask",
        help=(
            "Contour rendering source: mask computes boundaries from the refined mask store; persisted draws "
            "components/<component>/contours; auto prefers persisted and falls back to mask; "
            "compare draws both."
        ),
    )
    parser.add_argument(
        "--show-skeleton",
        action="store_true",
        help="Recompute and overlay subject-body skeleton pixels from the refined mask store for debug review.",
    )
    parser.add_argument(
        "--skeleton-style",
        choices=SKELETON_STYLE_CHOICES,
        default="underlay",
        help=(
            "Skeleton rendering mode: underlay draws raw skeleton beneath centerline; "
            "offset shifts skeleton points for overlap inspection; branches highlights "
            "skeleton pixels not selected by the longest-path centerline."
        ),
    )
    parser.add_argument(
        "--skeleton-offset-px",
        type=float,
        default=1.5,
        help="Pixel offset used by --skeleton-style offset.",
    )
    parser.add_argument(
        "--show-bspline",
        action="store_true",
        help="Draw components/subject_body/bspline_sample_xy when present.",
    )
    parser.add_argument(
        "--show-spline-control-points",
        action="store_true",
        help="Draw components/subject_body/bspline_control_points_xy when present.",
    )
    parser.add_argument(
        "--show-tail-samples",
        action="store_true",
        help="Draw components/subject_body/tail_sample_xy when present.",
    )
    parser.add_argument(
        "--show-tail-normals",
        action="store_true",
        help="Draw tail normals from components/subject_body/tail_normal_xy when present.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    rows = _parse_rows(args.rows, start_row=int(args.start_row), count=int(args.count))
    paths = export_subject_shape_overlays(
        args.zarr_path,
        output_dir=args.output_dir,
        shape_run=args.shape_run,
        refined_run=args.refined_run,
        crop_run=args.crop_run,
        rows=rows,
        use_crop_images=not bool(args.no_crop_images),
        dpi=int(args.dpi),
        contour_source=args.contour_source,
        show_skeleton=bool(args.show_skeleton),
        skeleton_style=args.skeleton_style,
        skeleton_offset_px=float(args.skeleton_offset_px),
        show_bspline=bool(args.show_bspline),
        show_tail_samples=bool(args.show_tail_samples),
        show_tail_normals=bool(args.show_tail_normals),
        show_spline_control_points=bool(args.show_spline_control_points),
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
