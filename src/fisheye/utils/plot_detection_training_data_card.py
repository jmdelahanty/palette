#!/usr/bin/env python3
"""Render plot PNGs from a detection training data-card JSON payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Optional, Sequence

import numpy as np


HISTOGRAM_NAMES = ("w_norm", "h_norm", "area_norm", "aspect_ratio")
GENOTYPE_COUNTS_PLOT_NAME = "genotype_counts"
DPF_HISTOGRAM_PLOT_NAME = "dpf_histogram"

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


def _parse_histogram(hist: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    edges_raw = hist.get("bin_edges")
    counts_raw = hist.get("counts")
    if not isinstance(edges_raw, Sequence) or not isinstance(counts_raw, Sequence):
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
) -> None:
    assert plt is not None
    edges, counts = _parse_histogram(hist)
    widths = np.diff(edges)
    centers = edges[:-1] + (widths / 2.0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(centers, counts, width=widths, color="#2E6F95", edgecolor="#123146", linewidth=0.8)
    x_limits = _histogram_focus_xlim(edges=edges, counts=counts)
    if x_limits is not None:
        ax.set_xlim(*x_limits)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
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


def _plot_genotype_counts(
    *,
    genotype_counts: Mapping[str, Any],
    output_path: Path,
) -> bool:
    assert plt is not None
    parsed = _parse_positive_count_mapping(genotype_counts)
    if not parsed:
        return False

    labels = [item[0] for item in parsed]
    counts = np.asarray([item[1] for item in parsed], dtype=np.float64)
    y_pos = np.arange(len(labels))

    fig_height = min(11.0, max(3.5, 0.45 * len(labels) + 1.6))
    fig, ax = plt.subplots(figsize=(9.5, fig_height))
    ax.barh(y_pos, counts, color="#4A7C59", edgecolor="#264A38", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title("Genotype Distribution")
    ax.set_xlabel("Dataset Count")
    ax.set_ylabel("Genotype")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def _subject_histogram_has_data(hist_payload: Any) -> bool:
    if not isinstance(hist_payload, Mapping):
        return False
    try:
        _, counts = _parse_histogram(hist_payload)
    except Exception:
        return False
    if counts.size == 0:
        return False
    return bool(np.any(counts > 0))


def _histogram_focus_xlim(*, edges: np.ndarray, counts: np.ndarray) -> Optional[tuple[float, float]]:
    if edges.size < 2 or counts.size == 0:
        return None
    lower = float(edges[0])
    upper = float(edges[-1])
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        return None

    occupied = np.flatnonzero(counts > 0)
    if occupied.size == 0:
        return lower, upper
    left_idx = int(occupied[0])
    right_idx = int(occupied[-1] + 1)
    left = float(edges[left_idx])
    right = float(edges[right_idx])
    if not np.isfinite(left) or not np.isfinite(right) or right <= left:
        return lower, upper

    padding = 0.05 * (right - left)
    focused_left = max(lower, left - padding)
    focused_right = min(upper, right + padding)
    if focused_right <= focused_left:
        return lower, upper
    return focused_left, focused_right


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
    if not isinstance(density_raw, Sequence):
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
    ax.set_title("Detection Center Heatmap")
    ax.set_xlabel("X Bin")
    ax.set_ylabel("Y Bin")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Density")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _expected_plot_paths(*, card_payload: Mapping[str, Any], output_dir: Path, prefix: str) -> list[Path]:
    expected: list[Path] = []
    histograms = card_payload.get("histograms_aggregate")
    if isinstance(histograms, Mapping):
        for hist_name in HISTOGRAM_NAMES:
            hist_payload = histograms.get(hist_name)
            if isinstance(hist_payload, Mapping):
                expected.append(output_dir / f"{prefix}.{hist_name}.png")

    spatial = card_payload.get("spatial_aggregate")
    if isinstance(spatial, Mapping):
        center_heatmap = spatial.get("center_heatmap")
        if isinstance(center_heatmap, Mapping):
            expected.append(output_dir / f"{prefix}.center_heatmap.png")

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


def generate_detection_training_data_card_plots(
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

    histograms = card_payload.get("histograms_aggregate")
    if isinstance(histograms, Mapping):
        for hist_name in HISTOGRAM_NAMES:
            hist_payload = histograms.get(hist_name)
            if not isinstance(hist_payload, Mapping):
                continue
            output_path = output_dir / f"{prefix}.{hist_name}.png"
            _plot_histogram(
                hist=hist_payload,
                title=f"{hist_name} Distribution",
                xlabel=hist_name,
                output_path=output_path,
            )
            generated.append(output_path)

    spatial = card_payload.get("spatial_aggregate")
    if isinstance(spatial, Mapping):
        center_heatmap = spatial.get("center_heatmap")
        if isinstance(center_heatmap, Mapping):
            output_path = output_dir / f"{prefix}.center_heatmap.png"
            _plot_center_heatmap(
                heatmap=center_heatmap,
                output_path=output_path,
                heatmap_bin_factor=int(heatmap_bin_factor),
            )
            generated.append(output_path)

    genotype_counts = card_payload.get("genotype_counts")
    if isinstance(genotype_counts, Mapping):
        output_path = output_dir / f"{prefix}.{GENOTYPE_COUNTS_PLOT_NAME}.png"
        if _plot_genotype_counts(genotype_counts=genotype_counts, output_path=output_path):
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
        )
        generated.append(output_path)

    return generated


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, required=True, help="Detection training data-card JSON path.")
    parser.add_argument(
        "--output-dir",
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
        print(f"Detection training data-card plots: mode=dry-run output_dir={output_dir} prefix={prefix}")
        return 0

    if args.view and (not args.force) and expected_paths and len(existing_paths) == len(expected_paths):
        open_errors = _open_paths(existing_paths)
        print(
            "Detection training data-card plots: mode=view-existing "
            f"opened={len(existing_paths)} output_dir={output_dir} open_errors={open_errors}"
        )
        return 1 if open_errors else 0

    try:
        generated = generate_detection_training_data_card_plots(
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
            "Detection training data-card plots: mode=apply+view "
            f"generated={len(generated)} output_dir={output_dir} open_errors={open_errors}"
        )
        return 1 if open_errors else 0

    print(
        "Detection training data-card plots: mode=apply "
        f"generated={len(generated)} output_dir={output_dir}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
