"""Render subject-shape centerline and tail-anchor overlays."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import zarr
from skimage.measure import find_contours

from fisheye.shared.crop_image_source import CropImageSource
from fisheye.utils.zarr_io import open_zarr_root


DEFAULT_OUTPUT_DIR = Path("/tmp/palette_subject_shape_overlays")


@dataclass(frozen=True)
class SubjectShapeOverlayContext:
    root: zarr.Group
    shape_run_name: str
    shape_group: zarr.Group
    refined_run_name: str
    refined_group: zarr.Group
    label_map: dict[str, int]
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
    if idx is None or "masks_roi" not in ctx.refined_group:
        return None
    masks = ctx.refined_group["masks_roi"]
    if int(row) < 0 or int(row) >= int(masks.shape[0]):
        raise IndexError(f"Row {row} out of range for masks_roi with {masks.shape[0]} rows.")
    return np.asarray(masks[int(row), int(idx)], dtype=np.uint8) > 0


def _base_image(ctx: SubjectShapeOverlayContext, row: int, subject_body: np.ndarray | None) -> tuple[np.ndarray, str]:
    if ctx.crop_source is not None and 0 <= int(row) < ctx.crop_source.total_rois:
        try:
            return _normalize_image(ctx.crop_source[int(row)]), f"crop:{ctx.crop_source.crop_run_name}"
        except Exception:
            pass
    if subject_body is not None:
        return np.asarray(subject_body, dtype=np.float32) * 0.55, "subject_body_mask"
    raise ValueError("No crop image or subject-body mask available for overlay background.")


def _plot_contours(ax: plt.Axes, mask: np.ndarray | None, *, color: str, linewidth: float, label: str) -> None:
    if mask is None or not np.any(mask):
        return
    for contour in find_contours(np.asarray(mask, dtype=np.float32), level=0.5):
        if int(contour.shape[0]) < 2:
            continue
        ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=linewidth, label=label)
        label = "_nolegend_"


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
) -> plt.Figure:
    row = int(row)
    body = _mask_for(ctx, row, "subject_body")
    swim = _mask_for(ctx, row, "swim_bladder")
    eye_left = _mask_for(ctx, row, "eye_left")
    eye_right = _mask_for(ctx, row, "eye_right")
    base, base_source = _base_image(ctx, row, body)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(base, cmap="gray", interpolation="nearest")
    _plot_contours(ax, body, color="white", linewidth=1.2, label="body contour")
    _plot_contours(ax, swim, color="#00bcd4", linewidth=1.5, label="swim bladder")
    _plot_contours(ax, eye_left, color="#ef5350", linewidth=1.2, label="eye left")
    _plot_contours(ax, eye_right, color="#42a5f5", linewidth=1.2, label="eye right")

    shape = ctx.shape_group
    centerline_valid = bool(_row_value(shape, "components/subject_body/centerline_valid", row, False))
    _plot_centerline(
        ax,
        _row_value(shape, "components/subject_body/centerline_xy", row, None),
        valid=centerline_valid,
    )
    _plot_body_frame(ax, shape, row)
    _plot_point(
        ax,
        _row_value(shape, "components/swim_bladder/caudal_contour_point_xy", row, None),
        color="#ffeb3b",
        marker="*",
        label="caudal swim anchor",
        size=110,
    )
    _plot_point(
        ax,
        _row_value(shape, "components/subject_body/head_endpoint_xy", row, None),
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
    tail_reason = _decode_reason(_row_value(shape, "components/subject_body/tail_base_failure_reason_bytes", row, []))
    anchor_reason = _decode_reason(
        _row_value(shape, "components/swim_bladder/caudal_contour_failure_reason_bytes", row, [])
    )
    body_len = _row_value(shape, "components/subject_body/body_arclength_px", row, np.nan)
    tail_len = _row_value(shape, "components/subject_body/tail_segment_arclength_px", row, np.nan)
    title = f"Subject shape overlay | row {row}"
    if frame is not None:
        title += f" | frame {int(frame)}"
    ax.set_title(title)
    text = "\n".join(
        [
            f"shape: {ctx.shape_run_name}",
            f"refined: {ctx.refined_run_name}",
            f"background: {base_source}",
            f"centerline: {centerline_valid} ({body_reason or 'n/a'})",
            f"tail_base: {bool(_row_value(shape, 'components/subject_body/tail_base_valid', row, False))} ({tail_reason or 'n/a'})",
            f"caudal_anchor: {bool(_row_value(shape, 'components/swim_bladder/caudal_contour_valid', row, False))} ({anchor_reason or 'n/a'})",
            f"body_len_px: {float(body_len):.2f}" if np.isfinite(float(body_len)) else "body_len_px: n/a",
            f"tail_len_px: {float(tail_len):.2f}" if np.isfinite(float(tail_len)) else "tail_len_px: n/a",
        ]
    )
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
        fig = render_subject_shape_overlay(ctx, row=int(row))
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
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
