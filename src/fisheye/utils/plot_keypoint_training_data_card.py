#!/usr/bin/env python3
"""Render plot PNGs from a keypoint training data-card JSON payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Optional, Sequence

import numpy as np


USABLE_RATE_PLOT_NAME = "usable_rate_distribution"
TRIANGLE_AREA_PLOT_NAME = "triangle_area_distribution"
MIN_ANGLE_PLOT_NAME = "min_angle_distribution"
HEADING_PLOT_NAME = "heading_distribution"
LANDMARK_HEATMAP_PANEL_PLOT_NAME = "landmark_heatmap_panel"
EDGE_LENGTH_NORM_PANEL_PLOT_NAME = "edge_length_norm_panel"
GENOTYPE_COUNTS_PLOT_NAME = "genotype_counts"
DPF_HISTOGRAM_PLOT_NAME = "dpf_histogram"

HISTOGRAM_PLOT_SPECS: tuple[tuple[str, str, str, tuple[tuple[str, ...], ...]], ...] = (
    (
        USABLE_RATE_PLOT_NAME,
        "Usable Keypoint Rate Distribution",
        "Usable keypoint rate",
        (
            ("quality", "usable_keypoints_rate_histogram"),
            ("quality", "usable_rate_histogram"),
            ("quality", "usable_keypoints_rate_distribution"),
            ("usable_keypoints_rate_histogram",),
            ("usable_rate_histogram",),
        ),
    ),
    (
        TRIANGLE_AREA_PLOT_NAME,
        "Triangle Area Distribution",
        "Triangle area",
        (
            ("geometry", "triangle_area", "histogram"),
            ("geometry", "triangle_area_histogram"),
            ("geometry", "triangle_area_stats", "histogram"),
            ("triangle_area_histogram",),
        ),
    ),
    (
        MIN_ANGLE_PLOT_NAME,
        "Minimum Angle Distribution",
        "Minimum angle (deg)",
        (
            ("geometry", "min_angle", "histogram"),
            ("geometry", "min_angle_histogram"),
            ("geometry", "min_angle_stats", "histogram"),
            ("min_angle_histogram",),
        ),
    ),
    (
        HEADING_PLOT_NAME,
        "Heading Distribution",
        "Heading (deg)",
        (
            ("geometry", "heading", "histogram"),
            ("geometry", "heading_histogram"),
            ("geometry", "heading_stats", "histogram"),
            ("heading_histogram",),
        ),
    ),
)

LANDMARK_HEATMAP_PATHS: tuple[tuple[str, ...], ...] = (
    ("spatial", "landmark_center_heatmaps"),
    ("spatial", "center_heatmaps"),
    ("spatial", "landmark_heatmaps"),
    ("landmark_center_heatmaps",),
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
    x_limits = _histogram_focus_xlim(edges=edges, counts=counts)
    if x_limits is not None:
        ax.set_xlim(*x_limits)
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


def _resolve_histogram_payload(
    card_payload: Mapping[str, Any],
    candidate_paths: Sequence[Sequence[str]],
) -> Optional[Mapping[str, Any]]:
    for path in candidate_paths:
        payload = _mapping_at_path(card_payload, path)
        if isinstance(payload, Mapping):
            return payload
    return None


def _resolve_edge_length_histograms(card_payload: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    graph_metrics = _mapping_at_path(card_payload, ("skeleton_graph_metrics", "edge_length_norm_stats"))
    if not isinstance(graph_metrics, Mapping):
        return []
    resolved: list[tuple[str, Mapping[str, Any]]] = []
    for key in sorted(graph_metrics.keys(), key=lambda item: str(item).lower()):
        entry = graph_metrics.get(key)
        if not isinstance(entry, Mapping):
            continue
        hist = entry.get("histogram")
        if not isinstance(hist, Mapping):
            continue
        label = _normalize_text(entry.get("alias")) or _normalize_text(key) or "edge"
        resolved.append((label, hist))
    return resolved


def _is_heatmap_payload(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    return "grid_h" in value and "grid_w" in value and "density" in value


def _heatmap_sort_key(label: str) -> tuple[int, Any]:
    text = str(label).strip()
    try:
        return (0, int(text))
    except Exception:
        return (1, text.lower())


def _resolve_landmark_label(default_label: str, payload: Mapping[str, Any]) -> str:
    for key in (
        "alias",
        "label",
        "name",
        "landmark_alias",
        "landmark_label",
        "landmark_name",
    ):
        text = _normalize_text(payload.get(key))
        if text is not None:
            return text
    return default_label


def _extract_landmark_heatmap_record(
    *,
    default_label: str,
    payload: Any,
) -> Optional[tuple[str, Mapping[str, Any]]]:
    if _is_heatmap_payload(payload):
        assert isinstance(payload, Mapping)
        return _resolve_landmark_label(default_label, payload), payload
    if not isinstance(payload, Mapping):
        return None
    nested_payload = payload.get("center_heatmap")
    if nested_payload is None:
        nested_payload = payload.get("heatmap")
    if not _is_heatmap_payload(nested_payload):
        return None
    assert isinstance(nested_payload, Mapping)
    return _resolve_landmark_label(default_label, payload), nested_payload


def _resolve_landmark_heatmaps(card_payload: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    for path in LANDMARK_HEATMAP_PATHS:
        raw_payload = _value_at_path(card_payload, path)
        parsed: list[tuple[str, Mapping[str, Any]]] = []

        if _is_heatmap_payload(raw_payload):
            record = _extract_landmark_heatmap_record(default_label="landmark_0", payload=raw_payload)
            if record is not None:
                parsed.append(record)
        elif isinstance(raw_payload, Mapping):
            for key in sorted(raw_payload.keys(), key=lambda item: _heatmap_sort_key(str(item))):
                record = _extract_landmark_heatmap_record(
                    default_label=f"landmark_{key}",
                    payload=raw_payload[key],
                )
                if record is not None:
                    parsed.append(record)
        elif isinstance(raw_payload, Sequence) and not isinstance(raw_payload, (str, bytes, bytearray)):
            for index, item in enumerate(raw_payload):
                record = _extract_landmark_heatmap_record(
                    default_label=f"landmark_{index}",
                    payload=item,
                )
                if record is not None:
                    parsed.append(record)

        if parsed:
            return parsed

    return []


def _parse_heatmap_grid(heatmap: Mapping[str, Any], heatmap_bin_factor: int) -> np.ndarray:
    grid_h = int(heatmap["grid_h"])
    grid_w = int(heatmap["grid_w"])
    density_raw = heatmap["density"]
    if not isinstance(density_raw, Sequence) or isinstance(density_raw, (str, bytes, bytearray)):
        raise ValueError("Landmark center heatmap density must be a sequence.")

    density = np.asarray([float(v) for v in density_raw], dtype=np.float64)
    if density.size != grid_h * grid_w:
        raise ValueError("Landmark center heatmap density length does not match grid size.")

    grid = density.reshape((grid_h, grid_w))
    grid = _coarsen_heatmap_grid(grid, int(heatmap_bin_factor))
    grid_sum = float(np.sum(grid))
    if grid_sum > 0:
        grid = grid / grid_sum
    return grid


def _plot_landmark_heatmap_panel(
    *,
    landmark_heatmaps: Sequence[tuple[str, Mapping[str, Any]]],
    output_path: Path,
    heatmap_bin_factor: int,
) -> None:
    assert plt is not None

    parsed: list[tuple[str, np.ndarray]] = []
    for raw_label, heatmap_payload in landmark_heatmaps:
        label = _normalize_text(raw_label) or "landmark"
        try:
            grid = _parse_heatmap_grid(heatmap_payload, int(heatmap_bin_factor))
        except Exception:
            continue
        parsed.append((label, grid))

    if not parsed:
        raise ValueError("No valid landmark center heatmaps available for plotting.")

    panel_count = len(parsed)
    cols = int(min(4, max(1, np.ceil(np.sqrt(panel_count)))))
    rows = int(np.ceil(panel_count / cols))

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3.0 * cols + 1.0, 2.8 * rows + 0.6),
        squeeze=False,
    )

    color_source = None
    for index, (label, grid) in enumerate(parsed):
        ax = axes.flat[index]
        color_source = ax.imshow(grid, origin="lower", cmap="viridis", aspect="auto")
        ax.set_title(label, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    for index in range(panel_count, rows * cols):
        axes.flat[index].axis("off")

    if color_source is not None:
        cbar = fig.colorbar(color_source, ax=axes.ravel().tolist(), fraction=0.025, pad=0.01)
        cbar.set_label("Normalized Density")

    fig.suptitle("Landmark Center Heatmaps", fontsize=12)
    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.04, top=0.90, wspace=0.18, hspace=0.24)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_edge_length_norm_panel(
    *,
    edge_histograms: Sequence[tuple[str, Mapping[str, Any]]],
    output_path: Path,
) -> None:
    assert plt is not None
    parsed: list[tuple[str, np.ndarray, np.ndarray]] = []
    for raw_label, hist in edge_histograms:
        label = _normalize_text(raw_label) or "edge"
        try:
            edges, counts = _parse_histogram(hist)
        except Exception:
            continue
        parsed.append((label, edges, counts))
    if not parsed:
        raise ValueError("No valid edge-length histograms available for plotting.")

    panel_count = len(parsed)
    cols = int(min(3, max(1, np.ceil(np.sqrt(panel_count)))))
    rows = int(np.ceil(panel_count / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4.2 * cols, 2.9 * rows + 0.4),
        squeeze=False,
    )

    for index, (label, edges, counts) in enumerate(parsed):
        ax = axes.flat[index]
        widths = np.diff(edges)
        centers = edges[:-1] + (widths / 2.0)
        ax.bar(centers, counts, width=widths, color="#6D9DC5", edgecolor="#2C4A63", linewidth=0.7)
        x_limits = _histogram_focus_xlim(edges=edges, counts=counts)
        if x_limits is not None:
            ax.set_xlim(*x_limits)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Norm length", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(axis="y", alpha=0.2)

    for index in range(panel_count, rows * cols):
        axes.flat[index].axis("off")

    fig.suptitle("Skeleton Edge Length (Normalized) Distributions", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _expected_plot_paths(*, card_payload: Mapping[str, Any], output_dir: Path, prefix: str) -> list[Path]:
    expected: list[Path] = []

    for plot_name, _title, _xlabel, candidate_paths in HISTOGRAM_PLOT_SPECS:
        hist_payload = _resolve_histogram_payload(card_payload, candidate_paths)
        if isinstance(hist_payload, Mapping):
            expected.append(output_dir / f"{prefix}.{plot_name}.png")

    if _resolve_landmark_heatmaps(card_payload):
        expected.append(output_dir / f"{prefix}.{LANDMARK_HEATMAP_PANEL_PLOT_NAME}.png")

    if _resolve_edge_length_histograms(card_payload):
        expected.append(output_dir / f"{prefix}.{EDGE_LENGTH_NORM_PANEL_PLOT_NAME}.png")

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


def generate_keypoint_training_data_card_plots(
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

    landmark_heatmaps = _resolve_landmark_heatmaps(card_payload)
    if landmark_heatmaps:
        output_path = output_dir / f"{prefix}.{LANDMARK_HEATMAP_PANEL_PLOT_NAME}.png"
        _plot_landmark_heatmap_panel(
            landmark_heatmaps=landmark_heatmaps,
            output_path=output_path,
            heatmap_bin_factor=int(heatmap_bin_factor),
        )
        generated.append(output_path)

    edge_histograms = _resolve_edge_length_histograms(card_payload)
    if edge_histograms:
        output_path = output_dir / f"{prefix}.{EDGE_LENGTH_NORM_PANEL_PLOT_NAME}.png"
        _plot_edge_length_norm_panel(
            edge_histograms=edge_histograms,
            output_path=output_path,
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
            integer_center_ticks=True,
        )
        generated.append(output_path)

    return generated


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", type=Path, required=True, help="Keypoint training data-card JSON path.")
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
        help="Coarsening factor for landmark heatmap bins (default: 2 for larger bins).",
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
        print(f"Keypoint training data-card plots: mode=dry-run output_dir={output_dir} prefix={prefix}")
        return 0

    if args.view and (not args.force) and expected_paths and len(existing_paths) == len(expected_paths):
        open_errors = _open_paths(existing_paths)
        print(
            "Keypoint training data-card plots: mode=view-existing "
            f"opened={len(existing_paths)} output_dir={output_dir} open_errors={open_errors}"
        )
        return 1 if open_errors else 0

    try:
        generated = generate_keypoint_training_data_card_plots(
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
            "Keypoint training data-card plots: mode=apply+view "
            f"generated={len(generated)} output_dir={output_dir} open_errors={open_errors}"
        )
        return 1 if open_errors else 0

    print(
        "Keypoint training data-card plots: mode=apply "
        f"generated={len(generated)} output_dir={output_dir}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
