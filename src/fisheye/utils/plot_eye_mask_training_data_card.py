#!/usr/bin/env python3
"""Render plot PNGs from an eye-mask training data-card JSON payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Optional, Sequence

import numpy as np


USABLE_RATE_PLOT_NAME = "usable_rate_distribution"
EYE_SEPARATION_PLOT_NAME = "eye_separation_distribution"
ELLIPSE_MAJOR_PLOT_NAME = "ellipse_major_distribution"
ELLIPSE_MINOR_PLOT_NAME = "ellipse_minor_distribution"
LEFT_AREA_PLOT_NAME = "left_area_distribution"
RIGHT_AREA_PLOT_NAME = "right_area_distribution"
UNION_AREA_PLOT_NAME = "union_area_distribution"
AREA_RATIO_PLOT_NAME = "left_right_area_ratio_distribution"
CENTER_HEATMAP_PLOT_NAME = "center_heatmap"
COMPOSITION_COUNTS_PLOT_NAME = "composition_counts"
PARITY_DELTA_PLOT_NAME = "parity_train_val_delta"
GENOTYPE_COUNTS_PLOT_NAME = "genotype_counts"
DPF_HISTOGRAM_PLOT_NAME = "dpf_histogram"

HISTOGRAM_PLOT_SPECS: tuple[tuple[str, str, str, tuple[tuple[str, ...], ...]], ...] = (
    (
        USABLE_RATE_PLOT_NAME,
        "Usable ROI-Pair Rate Distribution",
        "Successful ROI-pair rate",
        (
            ("quality", "successful_roi_pair_rate_histogram"),
            ("quality", "usable_rate_histogram"),
        ),
    ),
    (
        EYE_SEPARATION_PLOT_NAME,
        "Eye Separation Distribution",
        "Eye separation (p50)",
        (
            ("geometry", "eye_separation_p50_histogram"),
            ("geometry", "eye_separation_histogram"),
        ),
    ),
    (
        ELLIPSE_MAJOR_PLOT_NAME,
        "Ellipse Major-Axis Distribution",
        "Ellipse major axis (p50)",
        (
            ("geometry", "ellipse_major_p50_histogram"),
            ("geometry", "major_axis_p50_histogram"),
        ),
    ),
    (
        ELLIPSE_MINOR_PLOT_NAME,
        "Ellipse Minor-Axis Distribution",
        "Ellipse minor axis (p50)",
        (
            ("geometry", "ellipse_minor_p50_histogram"),
            ("geometry", "minor_axis_p50_histogram"),
        ),
    ),
    (
        LEFT_AREA_PLOT_NAME,
        "Left-Eye Area Distribution",
        "Left-eye area (p50 px)",
        (
            ("geometry", "left_area_p50_histogram"),
            ("geometry", "area_left_p50_histogram"),
            ("geometry", "left_eye_area_p50_histogram"),
        ),
    ),
    (
        RIGHT_AREA_PLOT_NAME,
        "Right-Eye Area Distribution",
        "Right-eye area (p50 px)",
        (
            ("geometry", "right_area_p50_histogram"),
            ("geometry", "area_right_p50_histogram"),
            ("geometry", "right_eye_area_p50_histogram"),
        ),
    ),
    (
        UNION_AREA_PLOT_NAME,
        "Union Area Distribution",
        "Union area (p50 px)",
        (
            ("geometry", "union_area_p50_histogram"),
            ("geometry", "area_union_p50_histogram"),
            ("geometry", "combined_area_p50_histogram"),
            ("geometry", "area_p50_histogram"),
        ),
    ),
    (
        AREA_RATIO_PLOT_NAME,
        "Left/Right Area Ratio Distribution",
        "Left/right area ratio (p50)",
        (
            ("geometry", "area_lr_ratio_p50_histogram"),
            ("geometry", "left_right_area_ratio_p50_histogram"),
            ("geometry", "area_ratio_left_right_p50_histogram"),
            ("geometry", "lr_area_ratio_p50_histogram"),
        ),
    ),
)

try:  # pragma: no cover - import fallback is environment-specific
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - exercised when matplotlib missing
    plt = None  # type: ignore[assignment]
    _MATPLOTLIB_IMPORT_ERROR = exc
else:
    _MATPLOTLIB_IMPORT_ERROR = None


def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        text = value.decode("utf-8", "ignore").strip()
    else:
        text = str(value).strip()
    return text or None


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _default_plot_dir(card_path: Path) -> Path:
    return card_path.parent / f"{card_path.stem}.plots"


def _default_prefix(card_payload: Mapping[str, Any], card_path: Path) -> str:
    return _normalize_text(card_payload.get("set_id")) or card_path.stem


def _value_at_path(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    current: Any = payload
    for part in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _mapping_at_path(payload: Mapping[str, Any], path: Sequence[str]) -> Optional[Mapping[str, Any]]:
    value = _value_at_path(payload, path)
    return value if isinstance(value, Mapping) else None


def _parse_histogram(hist: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    edges_raw = hist.get("bin_edges")
    counts_raw = hist.get("counts")
    if not isinstance(edges_raw, Sequence) or isinstance(edges_raw, (str, bytes, bytearray)):
        raise ValueError("Histogram payload missing bin_edges/counts sequence.")
    if not isinstance(counts_raw, Sequence) or isinstance(counts_raw, (str, bytes, bytearray)):
        raise ValueError("Histogram payload missing bin_edges/counts sequence.")
    edges = np.asarray([float(v) for v in edges_raw], dtype=np.float64)
    counts = np.asarray([float(v) for v in counts_raw], dtype=np.float64)
    if edges.size != counts.size + 1:
        raise ValueError("Histogram bin_edges length must be counts length + 1.")
    return edges, counts


def _plot_histogram(
    *,
    hist: Mapping[str, Any],
    title: str,
    xlabel: str,
    output_path: Path,
    integer_center_ticks: bool = False,
) -> None:
    assert plt is not None
    edges, counts = _parse_histogram(hist)
    widths = np.diff(edges)
    centers = edges[:-1] + (widths / 2.0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(centers, counts, width=widths, color="#2E6F95", edgecolor="#123146", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    if integer_center_ticks:
        ticks = _infer_integer_center_ticks(edges)
        if ticks is not None and ticks.size > 0:
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(int(v)) for v in ticks])
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _parse_positive_count_mapping(payload: Mapping[str, Any]) -> list[tuple[str, float]]:
    parsed: list[tuple[str, float]] = []
    for raw_label, raw_count in payload.items():
        label = _normalize_text(raw_label)
        count = _as_float(raw_count)
        if label is None or count is None or not np.isfinite(count) or count <= 0:
            continue
        parsed.append((label, float(count)))
    parsed.sort(key=lambda item: (-item[1], item[0].lower()))
    return parsed


def _plot_positive_count_mapping(
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    counts: Mapping[str, Any],
    output_path: Path,
) -> bool:
    assert plt is not None
    parsed = _parse_positive_count_mapping(counts)
    if not parsed:
        return False

    labels = [item[0] for item in parsed]
    values = np.asarray([item[1] for item in parsed], dtype=np.float64)
    y_pos = np.arange(len(labels))

    fig_height = min(11.0, max(3.5, 0.45 * len(labels) + 1.6))
    fig, ax = plt.subplots(figsize=(10.0, fig_height))
    ax.barh(y_pos, values, color="#4A7C59", edgecolor="#264A38", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def _subject_histogram_has_data(hist_payload: Any) -> bool:
    if not isinstance(hist_payload, Mapping):
        return False
    try:
        _edges, counts = _parse_histogram(hist_payload)
    except Exception:
        return False
    if counts.size == 0:
        return False
    return bool(np.any(counts > 0))


def _infer_integer_center_ticks(edges: np.ndarray) -> Optional[np.ndarray]:
    if edges.size < 2:
        return None
    widths = np.diff(edges)
    if widths.size == 0:
        return None
    if not np.all(np.isfinite(widths)):
        return None
    if not np.allclose(widths, widths[0], atol=1e-9, rtol=0.0):
        return None
    if not np.isclose(float(widths[0]), 1.0, atol=1e-9, rtol=0.0):
        return None
    centers = edges[:-1] + (widths / 2.0)
    rounded = np.round(centers)
    if not np.allclose(centers, rounded, atol=1e-9, rtol=0.0):
        return None
    return rounded.astype(np.float64)


def _coarsen_heatmap_grid(grid: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return grid
    grid_h, grid_w = grid.shape
    effective = int(min(factor, grid_h, grid_w))
    if effective <= 1:
        return grid
    trimmed_h = (grid_h // effective) * effective
    trimmed_w = (grid_w // effective) * effective
    if trimmed_h <= 0 or trimmed_w <= 0:
        return grid
    trimmed = grid[:trimmed_h, :trimmed_w]
    return trimmed.reshape(trimmed_h // effective, effective, trimmed_w // effective, effective).sum(axis=(1, 3))


def _plot_center_heatmap(
    *,
    heatmap: Mapping[str, Any],
    output_path: Path,
    heatmap_bin_factor: int,
) -> None:
    assert plt is not None
    grid_h = int(heatmap["grid_h"])
    grid_w = int(heatmap["grid_w"])
    density_raw = heatmap["density"]
    if not isinstance(density_raw, Sequence) or isinstance(density_raw, (str, bytes, bytearray)):
        raise ValueError("Center heatmap density must be a sequence.")

    density = np.asarray([float(v) for v in density_raw], dtype=np.float64)
    if density.size != grid_h * grid_w:
        raise ValueError("Center heatmap density length does not match grid size.")

    grid = density.reshape((grid_h, grid_w))
    grid = _coarsen_heatmap_grid(grid, int(heatmap_bin_factor))
    grid_sum = float(np.sum(grid))
    if grid_sum > 0:
        grid = grid / grid_sum

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(grid, origin="lower", cmap="viridis", aspect="auto")
    ax.set_title("Eye-Mask Coverage Heatmap")
    ax.set_xlabel("X Bin")
    ax.set_ylabel("Y Bin")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Density")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _resolve_histogram_payload(
    card_payload: Mapping[str, Any],
    candidate_paths: Sequence[Sequence[str]],
) -> Optional[Mapping[str, Any]]:
    for path in candidate_paths:
        payload = _mapping_at_path(card_payload, path)
        if isinstance(payload, Mapping):
            return payload
    return None


def _resolve_composition_counts(
    card_payload: Mapping[str, Any],
) -> tuple[Optional[str], Optional[Mapping[str, Any]]]:
    composition = card_payload.get("composition")
    if not isinstance(composition, Mapping):
        return None, None
    counts = composition.get("counts")
    if not isinstance(counts, Mapping):
        return None, None

    for field_name in (
        "camera_id",
        "rig_id",
        "arena_id",
        "method",
        "protocol_name",
    ):
        field_counts = counts.get(field_name)
        if isinstance(field_counts, Mapping) and _parse_positive_count_mapping(field_counts):
            return field_name, field_counts

    for raw_field_name, raw_counts in counts.items():
        if isinstance(raw_counts, Mapping) and _parse_positive_count_mapping(raw_counts):
            return _normalize_text(raw_field_name) or "composition", raw_counts

    return None, None


def _resolve_parity_metric_deltas(card_payload: Mapping[str, Any]) -> list[tuple[str, float]]:
    parity = card_payload.get("parity")
    if not isinstance(parity, Mapping):
        return []
    metrics = parity.get("metrics")
    if not isinstance(metrics, Mapping):
        return []

    parsed: list[tuple[str, float]] = []
    for raw_name, raw_payload in metrics.items():
        metric_name = _normalize_text(raw_name)
        if metric_name is None:
            continue
        delta = None
        if isinstance(raw_payload, Mapping):
            delta = _as_float(raw_payload.get("delta"))
            if delta is None:
                train = _as_float(raw_payload.get("train"))
                val = _as_float(raw_payload.get("val"))
                if train is not None and val is not None:
                    delta = abs(train - val)
        else:
            delta = _as_float(raw_payload)
        if delta is None or not np.isfinite(delta):
            continue
        parsed.append((metric_name, float(abs(delta))))

    parsed.sort(key=lambda item: (-item[1], item[0].lower()))
    return parsed


def _plot_parity_deltas(
    *,
    metric_deltas: Sequence[tuple[str, float]],
    output_path: Path,
) -> bool:
    assert plt is not None
    if not metric_deltas:
        return False

    labels = [item[0] for item in metric_deltas]
    deltas = np.asarray([item[1] for item in metric_deltas], dtype=np.float64)
    y_pos = np.arange(len(labels))

    fig_height = min(11.0, max(3.5, 0.45 * len(labels) + 1.6))
    fig, ax = plt.subplots(figsize=(10.0, fig_height))
    ax.barh(y_pos, deltas, color="#A65E2E", edgecolor="#5E3215", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title("Train/Val Parity Absolute Deltas")
    ax.set_xlabel("Absolute delta")
    ax.set_ylabel("Metric")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def _expected_plot_paths(*, card_payload: Mapping[str, Any], output_dir: Path, prefix: str) -> list[Path]:
    expected: list[Path] = []

    for plot_name, _title, _xlabel, candidate_paths in HISTOGRAM_PLOT_SPECS:
        hist_payload = _resolve_histogram_payload(card_payload, candidate_paths)
        if isinstance(hist_payload, Mapping):
            expected.append(output_dir / f"{prefix}.{plot_name}.png")

    spatial = card_payload.get("spatial")
    if isinstance(spatial, Mapping) and isinstance(spatial.get("center_heatmap"), Mapping):
        expected.append(output_dir / f"{prefix}.{CENTER_HEATMAP_PLOT_NAME}.png")

    composition_field, composition_counts = _resolve_composition_counts(card_payload)
    if composition_field is not None and isinstance(composition_counts, Mapping):
        expected.append(output_dir / f"{prefix}.{COMPOSITION_COUNTS_PLOT_NAME}.png")

    parity_deltas = _resolve_parity_metric_deltas(card_payload)
    if parity_deltas:
        expected.append(output_dir / f"{prefix}.{PARITY_DELTA_PLOT_NAME}.png")

    genotype_counts = card_payload.get("genotype_counts")
    if isinstance(genotype_counts, Mapping) and _parse_positive_count_mapping(genotype_counts):
        expected.append(output_dir / f"{prefix}.{GENOTYPE_COUNTS_PLOT_NAME}.png")

    dpf_histogram = card_payload.get("dpf_histogram")
    if _subject_histogram_has_data(dpf_histogram):
        expected.append(output_dir / f"{prefix}.{DPF_HISTOGRAM_PLOT_NAME}.png")

    return expected


def _open_paths(paths: Sequence[Path]) -> int:
    open_errors = 0
    for path in paths:
        try:
            completed = subprocess.run(["xdg-open", str(path)], check=False)
            if int(completed.returncode) != 0:
                open_errors += 1
        except Exception:
            open_errors += 1
    return open_errors


def generate_eye_mask_training_data_card_plots(
    *,
    card_payload: Mapping[str, Any],
    output_dir: Path,
    prefix: str,
    heatmap_bin_factor: int = 2,
) -> list[Path]:
    if plt is None:
        raise RuntimeError(f"matplotlib is required to generate plots: {_MATPLOTLIB_IMPORT_ERROR}")

    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    for plot_name, title, xlabel, candidate_paths in HISTOGRAM_PLOT_SPECS:
        hist_payload = _resolve_histogram_payload(card_payload, candidate_paths)
        if not isinstance(hist_payload, Mapping):
            continue
        output_path = output_dir / f"{prefix}.{plot_name}.png"
        _plot_histogram(
            hist=hist_payload,
            title=title,
            xlabel=xlabel,
            output_path=output_path,
        )
        generated.append(output_path)

    spatial = card_payload.get("spatial")
    if isinstance(spatial, Mapping):
        center_heatmap = spatial.get("center_heatmap")
        if isinstance(center_heatmap, Mapping):
            output_path = output_dir / f"{prefix}.{CENTER_HEATMAP_PLOT_NAME}.png"
            _plot_center_heatmap(
                heatmap=center_heatmap,
                output_path=output_path,
                heatmap_bin_factor=int(heatmap_bin_factor),
            )
            generated.append(output_path)

    composition_field, composition_counts = _resolve_composition_counts(card_payload)
    if composition_field is not None and isinstance(composition_counts, Mapping):
        output_path = output_dir / f"{prefix}.{COMPOSITION_COUNTS_PLOT_NAME}.png"
        if _plot_positive_count_mapping(
            title=f"Composition Distribution ({composition_field})",
            xlabel="Dataset Count",
            ylabel=composition_field,
            counts=composition_counts,
            output_path=output_path,
        ):
            generated.append(output_path)

    parity_deltas = _resolve_parity_metric_deltas(card_payload)
    if parity_deltas:
        output_path = output_dir / f"{prefix}.{PARITY_DELTA_PLOT_NAME}.png"
        if _plot_parity_deltas(metric_deltas=parity_deltas, output_path=output_path):
            generated.append(output_path)

    genotype_counts = card_payload.get("genotype_counts")
    if isinstance(genotype_counts, Mapping):
        output_path = output_dir / f"{prefix}.{GENOTYPE_COUNTS_PLOT_NAME}.png"
        if _plot_positive_count_mapping(
            title="Genotype Distribution",
            xlabel="Dataset Count",
            ylabel="Genotype",
            counts=genotype_counts,
            output_path=output_path,
        ):
            generated.append(output_path)

    dpf_histogram = card_payload.get("dpf_histogram")
    if _subject_histogram_has_data(dpf_histogram):
        assert isinstance(dpf_histogram, Mapping)
        output_path = output_dir / f"{prefix}.{DPF_HISTOGRAM_PLOT_NAME}.png"
        _plot_histogram(
            hist=dpf_histogram,
            title="DPF Distribution",
            xlabel="DPF at acquisition",
            output_path=output_path,
            integer_center_ticks=True,
        )
        generated.append(output_path)

    return generated


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--card",
        "--card-json",
        dest="card",
        type=Path,
        required=True,
        help="Eye-mask training data-card JSON path.",
    )
    parser.add_argument(
        "--output-dir",
        "--outdir",
        dest="output_dir",
        type=Path,
        help="Output plot directory (default: <card_stem>.plots next to card JSON).",
    )
    parser.add_argument("--prefix", type=str, help="Output filename prefix (default: set_id or card stem).")
    parser.add_argument(
        "--heatmap-bin-factor",
        type=int,
        default=2,
        help="Coarsening factor for center heatmap bins (default: 2 for larger bins).",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Open plot files using xdg-open. Uses existing plots when available.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate plot files even when existing files are present.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Plan plots but do not write PNG files.")
    args = parser.parse_args(argv)

    if args.view and args.dry_run:
        parser.error("--view cannot be combined with --dry-run.")
    if args.heatmap_bin_factor < 1:
        parser.error("--heatmap-bin-factor must be >= 1.")

    card_path = Path(args.card)
    if not card_path.exists():
        print(f"Data-card plotting failed: card not found: {card_path}")
        return 1

    try:
        payload = json.loads(card_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Data-card plotting failed: unable to parse JSON: {card_path} ({exc})")
        return 1
    if not isinstance(payload, Mapping):
        print(f"Data-card plotting failed: card payload is not a JSON object: {card_path}")
        return 1

    output_dir = Path(args.output_dir) if args.output_dir is not None else _default_plot_dir(card_path)
    prefix = str(args.prefix).strip() if args.prefix is not None else _default_prefix(payload, card_path)
    if not prefix:
        prefix = card_path.stem

    expected_paths = _expected_plot_paths(card_payload=payload, output_dir=output_dir, prefix=prefix)
    existing_paths = [path for path in expected_paths if path.exists()]

    if args.dry_run:
        print(f"Eye-mask training data-card plots: mode=dry-run output_dir={output_dir} prefix={prefix}")
        return 0

    if args.view and (not args.force) and expected_paths and len(existing_paths) == len(expected_paths):
        open_errors = _open_paths(existing_paths)
        print(
            "Eye-mask training data-card plots: mode=view-existing "
            f"opened={len(existing_paths)} output_dir={output_dir} open_errors={open_errors}"
        )
        return 1 if open_errors else 0

    try:
        generated = generate_eye_mask_training_data_card_plots(
            card_payload=payload,
            output_dir=output_dir,
            prefix=prefix,
            heatmap_bin_factor=int(args.heatmap_bin_factor),
        )
    except Exception as exc:
        print(f"Data-card plotting failed: {exc}")
        return 1

    if args.view:
        open_errors = _open_paths(generated)
        print(
            "Eye-mask training data-card plots: mode=apply+view "
            f"generated={len(generated)} output_dir={output_dir} open_errors={open_errors}"
        )
        return 1 if open_errors else 0

    print(
        "Eye-mask training data-card plots: mode=apply "
        f"generated={len(generated)} output_dir={output_dir}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
