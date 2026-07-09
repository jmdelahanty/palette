#!/usr/bin/env python3
"""Plot fixed-K samples from refined subject-mask component contours."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import zarr

from fisheye.shared.crop_image_source import CropImageSource


DEFAULT_COMPONENT_K = {
    "subject_body": 256,
    "swim_bladder": 96,
    "eye_left": 64,
    "eye_right": 64,
}


@dataclass(frozen=True)
class ContourSample:
    row_index: int
    component: str
    raw_points: np.ndarray
    sampled_points: np.ndarray
    valid: bool


@dataclass(frozen=True)
class RoiImageSource:
    crop_run_name: str
    roi_images: object
    source_crop_row_ids: np.ndarray
    source_kind: str = "roi_images"
    row_position_fallback: bool = False

    def image_for_refined_row(self, row_index: int) -> np.ndarray | None:
        row = int(row_index)
        if row < 0 or row >= int(self.source_crop_row_ids.shape[0]):
            return None
        crop_row = int(self.source_crop_row_ids[row])
        if crop_row < 0 or crop_row >= int(self.roi_images.shape[0]):
            return None
        return np.asarray(self.roi_images[crop_row])


def _nearest_vertex_distances(source_xy: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    source = np.asarray(source_xy, dtype=np.float32).reshape(-1, 2)
    target = np.asarray(target_xy, dtype=np.float32).reshape(-1, 2)
    source = source[np.isfinite(source).all(axis=1)]
    target = target[np.isfinite(target).all(axis=1)]
    if source.shape[0] == 0 or target.shape[0] == 0:
        return np.full((source.shape[0],), np.nan, dtype=np.float32)
    deltas = source[:, None, :] - target[None, :, :]
    distances = np.sqrt(np.sum(deltas * deltas, axis=2, dtype=np.float32), dtype=np.float32)
    return np.min(distances, axis=1).astype(np.float32, copy=False)


def _closed_polyline_perimeter(points_xy: np.ndarray) -> float:
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] < 2:
        return float("nan")
    closed = points
    if not np.allclose(closed[0], closed[-1]):
        closed = np.concatenate([closed, closed[:1]], axis=0)
    return float(np.sum(np.linalg.norm(np.diff(closed, axis=0), axis=1)))


def _bbox_diagonal(points_xy: np.ndarray) -> float:
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] == 0:
        return float("nan")
    extent = np.nanmax(points, axis=0) - np.nanmin(points, axis=0)
    return float(np.linalg.norm(extent))


def contour_similarity_metrics(
    raw_points: np.ndarray,
    sampled_points: np.ndarray,
    *,
    pixel_to_mm: float | None = None,
) -> dict[str, float | int | None]:
    """Approximate contour set similarity with nearest-vertex distances.

    This intentionally uses only the plotted vertices, so it is cheap and easy
    to interpret: values are ROI pixels. ``raw_to_sampled`` answers how far
    original contour vertices are from the sampled representation; this is the
    useful direction for deciding if lower K is visually/geometrically adequate.
    """

    raw = np.asarray(raw_points, dtype=np.float32).reshape(-1, 2)
    sampled = np.asarray(sampled_points, dtype=np.float32).reshape(-1, 2)
    raw = raw[np.isfinite(raw).all(axis=1)]
    sampled = sampled[np.isfinite(sampled).all(axis=1)]
    if raw.shape[0] == 0 or sampled.shape[0] == 0:
        return {
            "raw_vertices": int(raw.shape[0]),
            "sampled_vertices": int(sampled.shape[0]),
            "contour_perimeter_px": float("nan"),
            "contour_bbox_diagonal_px": float("nan"),
            "raw_to_sampled_mean_px": float("nan"),
            "raw_to_sampled_p95_px": float("nan"),
            "raw_to_sampled_max_px": float("nan"),
            "sampled_to_raw_mean_px": float("nan"),
            "sampled_to_raw_max_px": float("nan"),
            "symmetric_hausdorff_px": float("nan"),
            "raw_to_sampled_p95_fraction_of_perimeter": float("nan"),
            "raw_to_sampled_p95_fraction_of_bbox_diagonal": float("nan"),
            "pixel_to_mm": pixel_to_mm,
            "raw_to_sampled_p95_mm": float("nan"),
            "symmetric_hausdorff_mm": float("nan"),
        }
    raw_to_sampled = _nearest_vertex_distances(raw, sampled)
    sampled_to_raw = _nearest_vertex_distances(sampled, raw)
    raw_max = float(np.nanmax(raw_to_sampled))
    sampled_max = float(np.nanmax(sampled_to_raw))
    raw_p95 = float(np.nanpercentile(raw_to_sampled, 95.0))
    perimeter_px = _closed_polyline_perimeter(raw)
    bbox_diagonal_px = _bbox_diagonal(raw)
    scale = float(pixel_to_mm) if pixel_to_mm is not None and np.isfinite(float(pixel_to_mm)) and float(pixel_to_mm) > 0 else None
    return {
        "raw_vertices": int(raw.shape[0]),
        "sampled_vertices": int(sampled.shape[0]),
        "contour_perimeter_px": perimeter_px,
        "contour_bbox_diagonal_px": bbox_diagonal_px,
        "raw_to_sampled_mean_px": float(np.nanmean(raw_to_sampled)),
        "raw_to_sampled_p95_px": raw_p95,
        "raw_to_sampled_max_px": raw_max,
        "sampled_to_raw_mean_px": float(np.nanmean(sampled_to_raw)),
        "sampled_to_raw_max_px": sampled_max,
        "symmetric_hausdorff_px": max(raw_max, sampled_max),
        "raw_to_sampled_p95_fraction_of_perimeter": (
            raw_p95 / perimeter_px if np.isfinite(perimeter_px) and perimeter_px > 0 else float("nan")
        ),
        "raw_to_sampled_p95_fraction_of_bbox_diagonal": (
            raw_p95 / bbox_diagonal_px if np.isfinite(bbox_diagonal_px) and bbox_diagonal_px > 0 else float("nan")
        ),
        "pixel_to_mm": scale,
        "raw_to_sampled_p95_mm": raw_p95 * scale if scale is not None else float("nan"),
        "symmetric_hausdorff_mm": max(raw_max, sampled_max) * scale if scale is not None else float("nan"),
    }


def resample_closed_polyline(points_xy: np.ndarray, k: int) -> np.ndarray:
    """Arc-length resample a closed contour to exactly ``k`` points.

    ``points_xy`` is interpreted in ROI pixel coordinates. The output samples
    the closed contour uniformly over arc length with ``endpoint=False`` so the
    first point is not duplicated as the last fixed-K point.
    """

    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    sample_count = int(k)
    if sample_count <= 0:
        raise ValueError("k must be positive.")
    if points.shape[0] == 0:
        return np.full((sample_count, 2), np.nan, dtype=np.float32)
    if points.shape[0] == 1:
        return np.repeat(points.astype(np.float32, copy=False), sample_count, axis=0)

    closed = points
    if not np.allclose(closed[0], closed[-1]):
        closed = np.concatenate([closed, closed[:1]], axis=0)
    segment_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    perimeter = float(np.sum(segment_lengths))
    if not np.isfinite(perimeter) or perimeter <= 0.0:
        return np.repeat(points[:1].astype(np.float32, copy=False), sample_count, axis=0)

    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)]).astype(np.float64)
    targets = np.linspace(0.0, perimeter, num=sample_count, endpoint=False, dtype=np.float64)
    x = np.interp(targets, cumulative, closed[:, 0].astype(np.float64))
    y = np.interp(targets, cumulative, closed[:, 1].astype(np.float64))
    return np.stack([x, y], axis=1).astype(np.float32)


def parse_component_k(values: Sequence[str] | None) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for value in values or ():
        if "=" not in value:
            raise argparse.ArgumentTypeError(f"Expected COMPONENT=K, received {value!r}.")
        component, raw_k = value.split("=", 1)
        component = component.strip()
        if not component:
            raise argparse.ArgumentTypeError(f"Component name is empty in {value!r}.")
        try:
            k = int(raw_k)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid K in {value!r}.") from exc
        if k <= 0:
            raise argparse.ArgumentTypeError(f"K must be positive in {value!r}.")
        mapping[component] = k
    return mapping


def component_k(component: str, overrides: dict[str, int], default_k: int) -> int:
    if component in overrides:
        return int(overrides[component])
    if component in DEFAULT_COMPONENT_K:
        return int(DEFAULT_COMPONENT_K[component])
    return int(default_k)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive.")
    return parsed


def _available_components(run: zarr.Group) -> list[str]:
    components = run.get("components")
    if not isinstance(components, zarr.Group):
        return []
    out: list[str] = []
    for name in sorted(components.keys()):
        comp = components[name]
        contours = comp.get("contours") if isinstance(comp, zarr.Group) else None
        if isinstance(contours, zarr.Group) and all(key in contours for key in ("ptr", "len", "points_xy")):
            out.append(str(name))
    return out


def _resolve_run(root: zarr.Group, run_name: str | None) -> tuple[str, zarr.Group]:
    parent = root["refined_subject_masks_runs"]
    if run_name:
        if run_name not in parent:
            raise ValueError(f"refined_subject_masks_runs/{run_name} not found.")
        return run_name, parent[run_name]
    latest = (
        parent.attrs.get("refined_subject_mask_review_status_latest")
        or parent.attrs.get("latest_complete")
        or parent.attrs.get("latest")
    )
    if not latest:
        raise ValueError("No run supplied and refined_subject_masks_runs has no latest pointer.")
    latest = str(latest)
    if latest not in parent:
        raise ValueError(f"Latest refined subject-mask run {latest!r} not found.")
    return latest, parent[latest]


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


def _short_source_label(label: str, *, max_chars: int = 44) -> str:
    text = str(label)
    if len(text) <= int(max_chars):
        return text
    if "/" in text:
        prefix, suffix = text.rsplit("/", 1)
        prefix = prefix[: max(1, int(max_chars) - len(suffix) - 4)]
        return f"{prefix}.../{suffix}"
    return f"{text[: max(1, int(max_chars) - 3)]}..."


def _coerce_positive_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed) or parsed <= 0.0:
        return None
    return parsed


def _inverse_positive_float(value: object) -> float | None:
    parsed = _coerce_positive_float(value)
    if parsed is None:
        return None
    return 1.0 / parsed


def resolve_pixel_to_mm(root: zarr.Group) -> tuple[float | None, str | None]:
    """Resolve camera pixel-to-mm scale for ROI-local contour diagnostics."""

    analysis = root.get("analysis")
    calibration = analysis.get("calibration") if isinstance(analysis, zarr.Group) else None
    candidates: list[tuple[str, object]] = []
    if isinstance(calibration, zarr.Group):
        candidates.extend(
            [
                ("analysis/calibration.attrs[pixel_to_mm]", calibration.attrs.get("pixel_to_mm")),
                (
                    "analysis/calibration.attrs[pixels_per_mm_camera]^-1",
                    _inverse_positive_float(calibration.attrs.get("pixels_per_mm_camera")),
                ),
                (
                    "analysis/calibration.attrs[pixels_per_mm]^-1",
                    _inverse_positive_float(calibration.attrs.get("pixels_per_mm")),
                ),
            ]
        )
    candidates.extend(
        [
            ("root.attrs[pixel_to_mm]", root.attrs.get("pixel_to_mm")),
            (
                "root.attrs[pixels_per_mm_camera]^-1",
                _inverse_positive_float(root.attrs.get("pixels_per_mm_camera")),
            ),
            (
                "root.attrs[pixels_per_mm]^-1",
                _inverse_positive_float(root.attrs.get("pixels_per_mm")),
            ),
        ]
    )
    for source, value in candidates:
        scale = _coerce_positive_float(value)
        if scale is not None:
            return scale, source
    return None, None


def resolve_roi_image_source(
    root: zarr.Group,
    run: zarr.Group,
    *,
    zarr_path: Path | None = None,
    crop_run: str | None = None,
    image_array: str = "roi_images",
    image_source: str = "auto",
    allow_row_position_fallback: bool = False,
) -> RoiImageSource | None:
    if image_source == "none":
        return None
    if image_source not in {"auto", "crop"}:
        raise ValueError(f"Unsupported image_source: {image_source!r}.")

    resolved_crop_run = crop_run or run.attrs.get("source_crop_run") or run.attrs.get("crop_run")
    if not resolved_crop_run:
        if image_source == "crop":
            raise ValueError("No crop run supplied and refined run has no source_crop_run attr.")
        return None
    resolved_crop_run = str(resolved_crop_run)
    crop_parent = root.get("crop_runs")
    crop_group = crop_parent.get(resolved_crop_run) if isinstance(crop_parent, zarr.Group) else None
    if not isinstance(crop_group, zarr.Group):
        if image_source == "crop":
            raise ValueError(f"crop_runs/{resolved_crop_run} not found.")
        return None
    row_position_fallback = False
    if "source_crop_row_ids" in run:
        source_crop_row_ids = np.asarray(run["source_crop_row_ids"][:], dtype=np.int64)
    elif allow_row_position_fallback:
        if "masks_roi" in run:
            row_count = int(run["masks_roi"].shape[0])
        elif "roi_coordinates_full" in crop_group:
            row_count = int(crop_group["roi_coordinates_full"].shape[0])
        elif image_array in crop_group:
            row_count = int(crop_group[image_array].shape[0])
        else:
            row_count = 0
        source_crop_row_ids = np.arange(row_count, dtype=np.int64)
        row_position_fallback = True
    else:
        if image_source == "crop":
            raise ValueError("Refined run has no source_crop_row_ids; refusing row-position crop image fallback.")
        return None

    try:
        crop_source = CropImageSource.open(root, crop_run=resolved_crop_run, zarr_path=zarr_path)
        return RoiImageSource(
            crop_run_name=resolved_crop_run,
            roi_images=crop_source,
            source_crop_row_ids=source_crop_row_ids,
            source_kind=str(crop_source.roi_read_mode),
            row_position_fallback=row_position_fallback,
        )
    except Exception:
        if image_array not in crop_group:
            if image_source == "crop":
                raise ValueError(f"crop_runs/{resolved_crop_run} missing readable ROI pixels.")
            return None
        roi_images = crop_group[image_array]
        return RoiImageSource(
            crop_run_name=resolved_crop_run,
            roi_images=roi_images,
            source_crop_row_ids=source_crop_row_ids,
            source_kind=str(image_array),
            row_position_fallback=row_position_fallback,
        )


def _contour_points_for_rows(run: zarr.Group, component: str, rows: np.ndarray) -> list[np.ndarray]:
    contours = run["components"][component]["contours"]
    ptr = np.asarray(contours["ptr"][rows], dtype=np.int64)
    length = np.asarray(contours["len"][rows], dtype=np.int64)
    points = contours["points_xy"]
    out: list[np.ndarray] = []
    for offset, count in zip(ptr.tolist(), length.tolist(), strict=True):
        if int(offset) < 0 or int(count) <= 0:
            out.append(np.empty((0, 2), dtype=np.float32))
            continue
        out.append(np.asarray(points[int(offset) : int(offset) + int(count)], dtype=np.float32).reshape(-1, 2))
    return out


def _valid_rows_for_components(run: zarr.Group, components: Sequence[str]) -> np.ndarray:
    if not components:
        return np.empty((0,), dtype=np.int64)
    mask: np.ndarray | None = None
    row_count = int(run["source_crop_row_ids"].shape[0]) if "source_crop_row_ids" in run else int(run["masks_roi"].shape[0])
    for component in components:
        lengths = np.asarray(run["components"][component]["contours"]["len"][:], dtype=np.int64)
        valid = lengths > 0
        mask = valid if mask is None else (mask & valid)
    if mask is None:
        return np.arange(row_count, dtype=np.int64)
    return np.flatnonzero(mask).astype(np.int64, copy=False)


def select_rows(
    run: zarr.Group,
    components: Sequence[str],
    *,
    rows: Sequence[int] | None,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    if rows:
        return np.asarray([int(row) for row in rows], dtype=np.int64)
    valid_rows = _valid_rows_for_components(run, components)
    if int(valid_rows.shape[0]) == 0:
        raise ValueError(f"No rows have valid contours for all selected components: {list(components)!r}")
    rng = np.random.default_rng(int(seed))
    count = min(int(sample_count), int(valid_rows.shape[0]))
    selected = rng.choice(valid_rows, size=count, replace=False)
    return np.sort(selected.astype(np.int64, copy=False))


def build_contour_samples(
    run: zarr.Group,
    components: Sequence[str],
    rows: Sequence[int],
    *,
    component_k_overrides: dict[str, int] | None = None,
    default_k: int = 64,
) -> list[ContourSample]:
    overrides = dict(component_k_overrides or {})
    row_array = np.asarray(rows, dtype=np.int64).reshape(-1)
    samples: list[ContourSample] = []
    for component in components:
        k = component_k(component, overrides, default_k)
        raw_by_row = _contour_points_for_rows(run, component, row_array)
        for row_index, raw_points in zip(row_array.tolist(), raw_by_row, strict=True):
            valid = int(raw_points.shape[0]) >= 2
            sampled = resample_closed_polyline(raw_points, k)
            samples.append(
                ContourSample(
                    row_index=int(row_index),
                    component=str(component),
                    raw_points=raw_points,
                    sampled_points=sampled,
                    valid=bool(valid),
                )
            )
    return samples


def plot_samples(
    run: zarr.Group,
    samples: Sequence[ContourSample],
    *,
    rows: Sequence[int],
    components: Sequence[str],
    output: Path,
    roi_image_source: RoiImageSource | None = None,
    overlay_mask: bool = True,
    mask_alpha: float = 0.25,
    layout: str = "comparison",
    comparison_k: Sequence[int] | None = None,
    dpi: int = 160,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433

    row_values = [int(row) for row in rows]
    component_values = [str(component) for component in components]
    sample_by_key = {(sample.row_index, sample.component): sample for sample in samples}
    mask_labels = list(run.attrs.get("mask_labels") or [])
    if layout not in {"comparison", "overlay"}:
        raise ValueError("layout must be one of {'comparison', 'overlay'}.")
    extra_k = tuple(dict.fromkeys(int(k) for k in (comparison_k or ()) if int(k) > 0))
    panel_specs: list[tuple[str, str]] = []
    for component in component_values:
        if layout == "comparison":
            panel_specs.append((component, "original"))
            panel_specs.append((component, "sampled"))
            for _k in extra_k:
                panel_specs.append((component, f"sampled:{int(_k)}"))
        else:
            panel_specs.append((component, "overlay"))

    fig, axes = plt.subplots(
        len(row_values),
        len(panel_specs),
        figsize=(4.0 * len(panel_specs), 4.0 * len(row_values)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_pos, row_index in enumerate(row_values):
        base_image = roi_image_source.image_for_refined_row(row_index) if roi_image_source is not None else None
        base_source = (
            f"crop:{roi_image_source.crop_run_name}/{roi_image_source.source_kind}"
            if roi_image_source is not None and base_image is not None
            else "mask"
        )
        for col_pos, (component, panel_kind) in enumerate(panel_specs):
            ax = axes[row_pos][col_pos]
            sample = sample_by_key[(row_index, component)]
            raster_shape: tuple[int, int] | None = None
            if base_image is not None:
                base = _normalize_image(base_image)
                if base.ndim >= 2:
                    raster_shape = (int(base.shape[0]), int(base.shape[1]))
                ax.imshow(base, cmap="gray", interpolation="nearest", origin="upper")
            if overlay_mask and "masks_roi" in run and component in mask_labels:
                component_index = int(mask_labels.index(component))
                mask = np.asarray(run["masks_roi"][row_index, component_index], dtype=np.uint8)
                if raster_shape is None:
                    raster_shape = (int(mask.shape[0]), int(mask.shape[1]))
                    ax.imshow(mask, cmap="gray", alpha=0.45, interpolation="nearest", origin="upper")
                else:
                    visible_mask = np.ma.masked_where(mask <= 0, mask)
                    ax.imshow(
                        visible_mask,
                        cmap="autumn",
                        alpha=float(mask_alpha),
                        interpolation="nearest",
                        origin="upper",
                    )
            raw = sample.raw_points
            if raw.shape[0] > 0 and panel_kind in {"original", "overlay"}:
                closed = np.concatenate([raw, raw[:1]], axis=0) if raw.shape[0] > 1 else raw
                ax.plot(closed[:, 0], closed[:, 1], color="white", linewidth=1.2, alpha=0.95, label="original")
            if panel_kind.startswith("sampled:"):
                sampled_k = int(panel_kind.split(":", 1)[1])
                sampled = resample_closed_polyline(raw, sampled_k)
            else:
                sampled = sample.sampled_points
            if np.any(np.isfinite(sampled)) and panel_kind in {"sampled", "overlay"}:
                ax.plot(
                    sampled[:, 0],
                    sampled[:, 1],
                    "-o",
                    color="#d95f02",
                    linewidth=0.7,
                    markersize=0.8,
                    alpha=0.72,
                    markeredgewidth=0.0,
                )
            if np.any(np.isfinite(sampled)) and panel_kind.startswith("sampled:"):
                ax.plot(
                    sampled[:, 0],
                    sampled[:, 1],
                    "-o",
                    color="#1b9e77",
                    linewidth=0.7,
                    markersize=0.8,
                    alpha=0.72,
                    markeredgewidth=0.0,
                )
            if panel_kind == "original":
                title_kind = f"original n={raw.shape[0]}"
            elif panel_kind == "sampled":
                title_kind = f"sampled K={sampled.shape[0]}"
            elif panel_kind.startswith("sampled:"):
                title_kind = f"sampled K={sampled.shape[0]}"
            else:
                title_kind = f"overlay K={sampled.shape[0]}"
            ax.set_title(f"row {row_index} | {component}\n{title_kind}", fontsize=9)
            ax.text(
                0.02,
                0.02,
                _short_source_label(base_source),
                transform=ax.transAxes,
                color="white",
                fontsize=7,
                va="bottom",
                ha="left",
                bbox={"boxstyle": "round,pad=0.2", "fc": "black", "ec": "none", "alpha": 0.55},
            )
            ax.set_aspect("equal")
            if raster_shape is not None:
                height, width = raster_shape
                ax.set_xlim(-0.5, float(width) - 0.5)
                ax.set_ylim(float(height) - 0.5, -0.5)
            else:
                ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def _parse_rows(value: str | None) -> list[int] | None:
    if not value:
        return None
    rows = [int(part.strip()) for part in value.split(",") if part.strip()]
    return rows or None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zarr", required=True, type=Path, help="Analysis Zarr path.")
    parser.add_argument("--run", help="refined_subject_masks_runs/<run>; defaults to latest pointer.")
    parser.add_argument("--component", action="append", help="Component to plot. Repeatable; defaults to all contour components.")
    parser.add_argument("--component-k", action="append", default=[], help="Per-component sample count, e.g. subject_body=256.")
    parser.add_argument("--default-k", type=int, default=64)
    parser.add_argument("--rows", help="Comma-separated explicit row indices.")
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image-source", choices=("auto", "crop", "none"), default="auto")
    parser.add_argument("--crop-run", help="Override crop_runs/<run> used for ROI image underlays.")
    parser.add_argument("--image-array", default="roi_images", help="Crop-run array used for ROI image underlays.")
    parser.add_argument(
        "--layout",
        choices=("comparison", "overlay"),
        default="comparison",
        help=(
            "Plot layout. comparison renders original and sampled contours in separate panels; "
            "overlay draws both in one panel (default: comparison)."
        ),
    )
    parser.add_argument(
        "--comparison-k",
        type=positive_int,
        action="append",
        default=[],
        help="Additional fixed-K sampled contour panel to show in comparison layout. Repeatable, e.g. --comparison-k 128.",
    )
    parser.add_argument(
        "--allow-row-position-image-fallback",
        action="store_true",
        help="Allow refined row N to use crop image row N when source_crop_row_ids is missing.",
    )
    parser.add_argument("--no-overlay-mask", action="store_true")
    parser.add_argument("--mask-alpha", type=float, default=0.25)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    root = zarr.open_group(str(args.zarr), mode="r", use_consolidated=False)
    run_name, run = _resolve_run(root, args.run)
    components = tuple(args.component or _available_components(run))
    if not components:
        raise SystemExit("No contour components found.")
    overrides = parse_component_k(args.component_k)
    rows = select_rows(
        run,
        components,
        rows=_parse_rows(args.rows),
        sample_count=int(args.sample_count),
        seed=int(args.seed),
    )
    samples = build_contour_samples(
        run,
        components,
        rows,
        component_k_overrides=overrides,
        default_k=int(args.default_k),
    )
    roi_image_source = resolve_roi_image_source(
        root,
        run,
        crop_run=args.crop_run,
        image_array=str(args.image_array),
        image_source=str(args.image_source),
        allow_row_position_fallback=bool(args.allow_row_position_image_fallback),
    )
    plot_samples(
        run,
        samples,
        rows=rows.tolist(),
        components=components,
        output=args.output,
        roi_image_source=roi_image_source,
        overlay_mask=not bool(args.no_overlay_mask),
        mask_alpha=float(args.mask_alpha),
        layout=str(args.layout),
        comparison_k=tuple(int(k) for k in args.comparison_k),
        dpi=int(args.dpi),
    )
    if args.json:
        import json

        pixel_to_mm, pixel_to_mm_source = resolve_pixel_to_mm(root)
        similarity: list[dict[str, object]] = []
        sample_lookup = {(sample.row_index, sample.component): sample for sample in samples}
        for row_index in rows.tolist():
            for component in components:
                sample = sample_lookup[(int(row_index), str(component))]
                k_values = [
                    component_k(str(component), overrides, int(args.default_k)),
                    *[int(k) for k in args.comparison_k],
                ]
                for k_value in dict.fromkeys(k_values):
                    sampled = (
                        sample.sampled_points
                        if int(k_value) == int(sample.sampled_points.shape[0])
                        else resample_closed_polyline(sample.raw_points, int(k_value))
                    )
                    similarity.append(
                        {
                            "row_index": int(row_index),
                            "component": str(component),
                            "k": int(k_value),
                            **contour_similarity_metrics(
                                sample.raw_points,
                                sampled,
                                pixel_to_mm=pixel_to_mm,
                            ),
                        }
                    )

        print(
            json.dumps(
                {
                    "status": "ok",
                    "run": run_name,
                    "components": list(components),
                    "rows": rows.tolist(),
                    "component_k": {
                        component: component_k(component, overrides, int(args.default_k))
                        for component in components
                    },
                    "comparison_k": [int(k) for k in args.comparison_k],
                    "pixel_to_mm": pixel_to_mm,
                    "pixel_to_mm_source": pixel_to_mm_source,
                    "contour_similarity": similarity,
                    "image_source": (
                        None
                        if roi_image_source is None
                        else {
                            "crop_run": roi_image_source.crop_run_name,
                            "source_kind": roi_image_source.source_kind,
                            "row_position_fallback": bool(roi_image_source.row_position_fallback),
                        }
                    ),
                    "output": str(args.output),
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
