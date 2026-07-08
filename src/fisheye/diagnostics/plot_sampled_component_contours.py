#!/usr/bin/env python3
"""Plot fixed-K samples from refined subject-mask component contours."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import zarr


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
    overlay_mask: bool = True,
    dpi: int = 160,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433

    row_values = [int(row) for row in rows]
    component_values = [str(component) for component in components]
    sample_by_key = {(sample.row_index, sample.component): sample for sample in samples}
    mask_labels = list(run.attrs.get("mask_labels") or [])

    fig, axes = plt.subplots(
        len(row_values),
        len(component_values),
        figsize=(4.0 * len(component_values), 4.0 * len(row_values)),
        squeeze=False,
        constrained_layout=True,
    )
    for row_pos, row_index in enumerate(row_values):
        for col_pos, component in enumerate(component_values):
            ax = axes[row_pos][col_pos]
            sample = sample_by_key[(row_index, component)]
            if overlay_mask and "masks_roi" in run and component in mask_labels:
                component_index = int(mask_labels.index(component))
                mask = np.asarray(run["masks_roi"][row_index, component_index], dtype=np.uint8)
                ax.imshow(mask, cmap="gray", alpha=0.35, origin="upper")
            raw = sample.raw_points
            if raw.shape[0] > 0:
                closed = np.concatenate([raw, raw[:1]], axis=0) if raw.shape[0] > 1 else raw
                ax.plot(closed[:, 0], closed[:, 1], color="0.6", linewidth=0.8, label="raw")
            sampled = sample.sampled_points
            if np.any(np.isfinite(sampled)):
                ax.plot(sampled[:, 0], sampled[:, 1], "-o", color="#d95f02", linewidth=1.0, markersize=2.0)
            ax.set_title(f"row {row_index} / {component} / K={sampled.shape[0]}")
            ax.set_aspect("equal")
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
    parser.add_argument("--no-overlay-mask", action="store_true")
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
    plot_samples(
        run,
        samples,
        rows=rows.tolist(),
        components=components,
        output=args.output,
        overlay_mask=not bool(args.no_overlay_mask),
        dpi=int(args.dpi),
    )
    if args.json:
        import json

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
                    "output": str(args.output),
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
